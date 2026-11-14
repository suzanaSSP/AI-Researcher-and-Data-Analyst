import re, shutil, subprocess, pathlib, os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-3.8-flash"
MAX_ATTEMPTS = 3

def run_tests(d):
    r = subprocess.run(["pytest", "-x", "-q"], cwd=d,
                       capture_output=True, text=True, timeout=60)
    return r.returncode == 0, r.stdout + r.stderr

def ask_fix(code, tests, output):
    prompt = (f"This code fails its tests.\n\nCODE:\n{code}\n\nTESTS:\n{tests}\n\n"
              f"TEST OUTPUT:\n{output}\n\n"
              "Return the full corrected code in one ```python block. "
              "Do not change the tests.")
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    text = response.text or ""
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return match.group(1) if match else None

def fix_one(case_dir):
    work = pathlib.Path("work") / case_dir.name
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(case_dir, work)     # never edit the original
    for attempt in range(1, MAX_ATTEMPTS + 1):
        passed, output = run_tests(work)
        if passed:
            return "fixed", attempt - 1
        new_code = ask_fix((work / "buggy.py").read_text(),
                           (work / "test_buggy.py").read_text(), output)
        if new_code is None:
            continue
        (work / "buggy.py").write_text(new_code)
    passed, _ = run_tests(work)
    return ("fixed", MAX_ATTEMPTS) if passed else ("escalate", MAX_ATTEMPTS)

if __name__ == "__main__":
    for case in sorted(pathlib.Path("cases").iterdir()):
        status, tries = fix_one(case)
        print(f"{case.name}: {status} ({tries} LLM attempts)")