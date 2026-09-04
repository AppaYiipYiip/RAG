# app.py
import logging
from flask import Flask, render_template, request, jsonify
import stt
import tts
import nlp_to_sql
import answer_generator
import database
import llm_utils
import graph_generator
from schema_loader import get_schema, load_schema_metadata, merge_schema_metadata
import conversation_manager
from config import MAX_SQL_GENERATION_ATTEMPTS, LOG_LEVEL

app = Flask(__name__)

# Reject request bodies over 15 MB outright (recording.wav from a 30s clip is
# a few MB at most). This runs before the /transcribe duration check, so an
# oversized upload never gets fully read into memory in the first place.
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024

# Configure logging. LOG_LEVEL affects the root logger, so DEBUG also
# surfaces Flask/Werkzeug/transformers' own debug output, not just ours,
# that's expected and fine for a debugging session, just noisier.
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load schema once, always fresh from whichever database is configured via
# .env at startup, no table/column names are hardcoded anywhere in the
# pipeline. Optionally enriched with hand-written business context from
# schema_metadata.yaml if that file exists (see
# schema_metadata.example.yaml for the format); the app runs fine without
# it, this is a pure addition.
SCHEMA = get_schema()
SCHEMA_METADATA = load_schema_metadata()
SCHEMA = merge_schema_metadata(SCHEMA, SCHEMA_METADATA)
logger.info(f"Schema loaded: {len(SCHEMA['tables'])} tables.")

# Warm up both LLM roles at startup so the first real request isn't slow
# (Whisper is already loaded eagerly on import of stt). warm_up() is a
# no-op for any role configured as an API backend, there's nothing local
# to preload in that case.
llm_utils.warm_up("reasoning")
llm_utils.warm_up("chat")
logger.info("LLM roles warmed up.")

@app.route('/')
def index():
    return render_template('index.html')


def _generate_and_execute_sql(question, schema, history_text, max_attempts=MAX_SQL_GENERATION_ATTEMPTS):
    """
    Generate SQL and execute it. Two things can trigger another attempt,
    fed back into the next prompt so the model can self-correct:
      1. A hard failure: rejected as non-read-only, or a real execution
         error (e.g. a syntax error like SQL Server rejecting LIMIT).
      2. A soft failure: the query runs fine, but the model's own semantic
         self-check on the result flags it as not really answering the
         question (e.g. a suspiciously empty result).

    On the final attempt, a soft (semantic) failure is still returned
    rather than discarded, since it's the model's own opinion, not a
    guarantee, and showing the best available result beats erroring out.
    A hard failure on the final attempt is raised, preserving the original
    exception type so the caller can keep distinguishing a rejected query
    (ValueError) from a genuine execution error.
    """
    sql = None
    last_feedback = None
    last_exception = None

    for attempt in range(1, max_attempts + 1):
        try:
            sql = nlp_to_sql.text_to_sql(
                question, schema, history_text=history_text,
                previous_sql=sql, previous_error=last_feedback
            )
            df = database.execute_sql(sql)
        except Exception as e:
            last_feedback = str(e)
            last_exception = e
            logger.warning(f"SQL attempt {attempt}/{max_attempts} failed to execute: {e} | SQL: {sql}")
            continue

        verdict = nlp_to_sql.evaluate_sql_result(question, sql, df)
        if verdict["verdict"] == "OK" or attempt == max_attempts:
            if verdict["verdict"] != "OK":
                logger.info(
                    f"Using result despite a RETRY self-check after exhausting attempts: {verdict['reason']}"
                )
            elif attempt > 1:
                logger.info(f"SQL succeeded (and passed self-check) on attempt {attempt}/{max_attempts}")
            return sql, df

        logger.info(f"SQL attempt {attempt}/{max_attempts}: model flagged its own result: {verdict['reason']}")
        last_feedback = verdict["reason"]
        last_exception = None  # this was a soft signal, not a real error

    # Only reachable if every attempt raised a hard exception. The (sql, df)
    # tuple is never returned to the caller in this path, so the caller's
    # own `sql` variable would never get bound, attach the last attempted
    # SQL onto the exception itself so it can still be logged/reported.
    last_exception.sql_attempted = sql
    raise last_exception


@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    logger.info(f"Received question: {question}")

    action = conversation_manager.decide_action(question)
    logger.info(f"Action: {action}")

    # ----- CHAT: no SQL, just conversational response -----
    if action == "CHAT":
        answer = answer_generator.generate_chat_response(
            question,
            history=conversation_manager.state.history
        )
        # Note: current_sql/current_df are deliberately left untouched here,
        # so a follow-up after some chit-chat can still refer to the last
        # real query result.
        conversation_manager.state.current_question = question
        conversation_manager.state.current_answer = answer
        turn_id = conversation_manager.add_turn(question, answer, turn_type="chat")

        return jsonify({
            'sql': '',
            'columns': [],
            'rows': [],
            'answer': answer,
            'action': 'chat',
            'turn_id': turn_id
        })

    # ----- FOLLOW_UP: reuse the active dataset, no new SQL -----
    if action == "FOLLOW_UP" and conversation_manager.state.current_df is not None:
        df = conversation_manager.state.current_df
        sql = conversation_manager.state.current_sql or ""
        recent_history = conversation_manager.format_history(
            conversation_manager.state.history[-conversation_manager.MAX_CONTEXT_TURNS_ANSWER:]
        )
        answer = answer_generator.generate_answer(
            question,
            df,
            sql,
            previous_context=recent_history
        )
        conversation_manager.state.current_question = question
        conversation_manager.state.current_answer = answer
        turn_id = conversation_manager.add_turn(question, answer, sql=sql, df=df, turn_type="sql")

        return jsonify({
            'sql': sql,
            'columns': df.columns.tolist(),
            'rows': df.values.tolist(),
            'answer': answer,
            'action': 'follow_up',
            'turn_id': turn_id,
            'chart_worthy': graph_generator.is_chart_worthy(df)
        })

    # ----- NEW_QUERY: generate SQL informed by recent context, execute, then answer -----
    sql_history = conversation_manager.format_history(
        conversation_manager.state.history[-conversation_manager.MAX_CONTEXT_TURNS_SQL:]
    )
    try:
        sql, df = _generate_and_execute_sql(question, SCHEMA, sql_history)
        logger.info(f"Generated SQL: {sql}")
        logger.info(f"SQL execution successful. {len(df)} rows returned.")
    except ValueError as e:
        # Raised by database.execute_sql when the query isn't read-only,
        # and still true after every retry. sql, df were never bound here
        # (the exception happened before that assignment could complete),
        # so read the last attempted SQL back off the exception instead.
        sql = getattr(e, 'sql_attempted', None)
        logger.warning(f"SQL rejected after retries: {e} | SQL: {sql}")
        return jsonify({'error': str(e), 'sql': sql}), 400
    except Exception as e:
        sql = getattr(e, 'sql_attempted', None)
        logger.error(f"SQL generation/execution failed after retries: {e} | SQL: {sql}")
        return jsonify({'error': f'SQL execution failed: {str(e)}', 'sql': sql}), 500

    answer_history = conversation_manager.format_history(
        conversation_manager.state.history[-conversation_manager.MAX_CONTEXT_TURNS_ANSWER:]
    )
    answer = answer_generator.generate_answer(question, df, sql, previous_context=answer_history)

    # Store as the active dataset for any follow-ups.
    conversation_manager.state.current_question = question
    conversation_manager.state.current_sql = sql
    conversation_manager.state.current_df = df
    conversation_manager.state.current_answer = answer
    turn_id = conversation_manager.add_turn(question, answer, sql=sql, df=df, turn_type="sql")

    return jsonify({
        'sql': sql,
        'columns': df.columns.tolist(),
        'rows': df.values.tolist(),
        'answer': answer,
        'action': 'new_query',
        'turn_id': turn_id,
        'chart_worthy': graph_generator.is_chart_worthy(df)
    })




@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    audio_file = request.files['audio']
    audio_bytes = audio_file.read()
    try:
        text = stt.transcribe_audio_bytes(audio_bytes)
        logger.info(f"Transcription: {text}")
        return jsonify({'text': text})
    except ValueError as e:
        # Raised by stt.transcribe_audio_bytes for invalid/too-long audio.
        logger.warning(f"Audio rejected: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/tts', methods=['POST'])
def text_to_speech():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    try:
        audio_bytes = tts.synthesize_speech(text)
        return audio_bytes, 200, {'Content-Type': 'audio/mpeg'}
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/graph', methods=['POST'])
def generate_graph():
    data = request.get_json()
    turn_id = data.get('turn_id')
    if turn_id is None:
        return jsonify({'error': 'No turn_id provided'}), 400

    turn = conversation_manager.get_turn(turn_id)
    if turn is None:
        return jsonify({'error': "Ce résultat n'est plus disponible pour générer un graphique."}), 404

    df = turn.get('df')
    if df is None or df.empty:
        return jsonify({'error': 'Aucune donnée disponible pour générer un graphique.'}), 400

    try:
        result = graph_generator.generate_graph(
            question=turn['question'],
            answer=turn['answer'],
            sql=turn.get('sql') or '',
            df=df,
            schema=SCHEMA
        )
    except Exception as e:
        logger.error(f"Graph generation failed: {e}")
        return jsonify({'error': f'Graph generation failed: {str(e)}'}), 500

    if 'error' in result:
        # A graceful, expected failure (bad spec, unknown column, etc.), not a server error.
        return jsonify({'error': result['error']}), 422

    return jsonify(result)


@app.route('/clear', methods=['POST'])
def clear_conversation():
    conversation_manager.clear_state()
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)