"""
network_check.py

Run this BEFORE paying for anything, to check whether this laptop's
network even allows the traffic the Spaces plan needs. Costs nothing,
creates nothing on Hugging Face, just makes a couple of read-only HTTPS
requests to things that already exist.

What it checks:
  1. Can we reach huggingface.co at all? Needed later for git push when
     you deploy a Space, and it's the more "normal looking" of the two
     domains, some corporate filters allow well known sites like this
     while still blocking less familiar ones.
  2. Can we reach an existing, already-running Space's own URL? This is
     the domain that actually matters most: *.hf.space is what
     llm_utils._call_api() and stt._transcribe_api() will be calling at
     runtime, every single query, and it's a different domain than
     huggingface.co itself, so passing check 1 does not guarantee this
     one also passes.

Pre-filled below with black-forest-labs/FLUX.1-dev, a popular, actively
maintained public Space, confirmed live at the time this script was
written. If it ever turns out to be down whenever you run this, swap in
any other live Space: open https://huggingface.co/spaces in a browser,
click into anything shown as "Running", and copy its URL from the address
bar once it loads (it looks like https://<owner>-<name>.hf.space). It
does not need to be your own, any live Space proves whether the network
path is open or not.

Usage:
    python network_check.py
"""

import os
import requests

TEST_SPACE_URL = "https://black-forest-labs-flux-1-dev.hf.space"


def check(name, url, timeout=10):
    print(f"\n--- {name} ---")
    print(f"GET {url}")
    try:
        resp = requests.get(url, timeout=timeout)
        print(f"OK, got a response back: status code {resp.status_code}")
        print("A non-200 status here still counts as a pass, what matters is that a")
        print("response came back at all rather than a timeout, reset, or proxy block.")
        return True
    except requests.exceptions.ProxyError as e:
        print(f"FAILED (proxy error): {e}")
        print("This laptop likely needs an explicit proxy configured for outbound HTTPS.")
        return False
    except requests.exceptions.SSLError as e:
        print(f"FAILED (SSL/TLS error): {e}")
        print("Possibly a corporate TLS-inspection proxy interfering, worth asking IT about.")
        return False
    except requests.exceptions.ConnectTimeout:
        print("FAILED (connection timed out)")
        print("This domain is likely blocked outright by a firewall.")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"FAILED (connection error): {e}")
        return False


def check_post(name, url, timeout=15):
    """
    Same idea as check(), but with an actual POST body, since that's what
    every real call in this app sends, never a plain GET. Any status code
    still counts as a pass here too, a 404 or 405 on this exact path is
    fine, what's being tested is whether a POST with a JSON body reaches
    the server at all, not whether this specific path does anything.
    """
    print(f"\n--- {name} ---")
    print(f"POST {url}")
    try:
        resp = requests.post(url, json={"network_check": True}, timeout=timeout)
        print(f"OK, got a response back: status code {resp.status_code}")
        print("Any status code here still counts as a pass, same reasoning as the GET check.")
        return True
    except requests.exceptions.ProxyError as e:
        print(f"FAILED (proxy error): {e}")
        return False
    except requests.exceptions.SSLError as e:
        print(f"FAILED (SSL/TLS error): {e}")
        return False
    except requests.exceptions.ConnectTimeout:
        print("FAILED (connection timed out)")
        print("POST specifically isn't getting through even if GET did, some proxies inspect")
        print("or block request bodies differently than they treat plain page loads.")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"FAILED (connection error): {e}")
        return False


print("Checking for proxy environment variables already set on this machine...")
proxy_vars = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
found_any = False
for var in proxy_vars:
    val = os.environ.get(var)
    if val:
        print(f"  {var} = {val}")
        found_any = True
if not found_any:
    print("  none found, this network likely expects direct outbound HTTPS, no proxy setup")

hf_ok = check("huggingface.co (needed for git push and model downloads)", "https://huggingface.co")
space_ok = check("a live Space, GET (*.hf.space, the domain actual inference calls use)", TEST_SPACE_URL)
post_ok = check_post("the same Space, POST with a body (the actual traffic pattern your app sends)", TEST_SPACE_URL)

print("\n--- Summary ---")
if hf_ok and space_ok and post_ok:
    print("All three checks passed, including POST. This is as much confidence as a test")
    print("against someone else's Space can give you, the only things left untested are your")
    print("own future Space's exact hostname and a slower, longer-held call, neither of which")
    print("can exist yet, both are virtually always fine if these three passed.")
elif hf_ok and space_ok and not post_ok:
    print("GET works but POST specifically does not. This is the one worth taking straight to")
    print("IT before paying for anything, some proxies allow ordinary page loads while")
    print("inspecting or blocking POST bodies, and every real call this app makes is a POST.")
elif hf_ok and not space_ok:
    print("huggingface.co works but *.hf.space does not. Worth flagging that second domain")
    print("specifically to IT if this laptop is managed, since it's the one the reasoning,")
    print("chat, and STT calls actually depend on at runtime, not huggingface.co itself.")
else:
    print("Neither worked. This network likely restricts outbound HTTPS to an allow-list that")
    print("does not include these domains yet, worth confirming with IT before paying for PRO.")