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
from schema_loader import get_schema
import conversation_manager

app = Flask(__name__)

# Reject request bodies over 15 MB outright (recording.wav from a 30s clip is
# a few MB at most). This runs before the /transcribe duration check, so an
# oversized upload never gets fully read into memory in the first place.
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load schema once
SCHEMA = get_schema()
logger.info(f"Schema loaded: {len(SCHEMA['tables'])} tables.")

# Warm up the LLM at startup (Whisper is already loaded eagerly on import of stt).
# Both models are small enough that eager loading is fine here.
llm_utils.get_llm()
logger.info("LLM model loaded.")

@app.route('/')
def index():
    return render_template('index.html')


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
            'turn_id': turn_id
        })

    # ----- NEW_QUERY: generate SQL informed by recent context, execute, then answer -----
    sql_history = conversation_manager.format_history(
        conversation_manager.state.history[-conversation_manager.MAX_CONTEXT_TURNS_SQL:]
    )
    try:
        sql = nlp_to_sql.text_to_sql(question, SCHEMA, history_text=sql_history)
        logger.info(f"Generated SQL: {sql}")
    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        return jsonify({'error': f'SQL generation failed: {str(e)}'}), 500

    try:
        df = database.execute_sql(sql)
        logger.info(f"SQL execution successful. {len(df)} rows returned.")
    except ValueError as e:
        # Raised by database.execute_sql when the query isn't read-only.
        logger.warning(f"SQL rejected: {e} | SQL: {sql}")
        return jsonify({'error': str(e), 'sql': sql}), 400
    except Exception as e:
        logger.error(f"SQL execution failed: {e} | SQL: {sql}")
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
        'turn_id': turn_id
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
            df=df
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