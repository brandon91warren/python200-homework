from dotenv import load_dotenv
from openai import OpenAI
import json

# --- Task 1: Setup and System Prompt ---

load_dotenv()
client = OpenAI()


def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content


YOUR_SYSTEM_PROMPT = """
You are a job application coach helping career changers translate their past experience into strong job application materials.

You can help with resume bullet points, cover letter openings, interview preparation, and job application questions.
Stay focused on job application materials and career communication.
Do not write anything that sounds dishonest or invent experience, credentials, numbers, companies, or achievements the user did not provide.
Always remind the user to review and edit your output before submitting it anywhere.
Acknowledge that you may not know every specific industry norm, so the user should use their own judgment before using your advice.
Use a supportive, practical, and professional tone.
"""

# I made the system prompt specific to career changers because this project is about helping people translate past experience into new career language.
# I also told the model not to invent achievements because resume and cover letter writing should stay honest and accurate.


# --- Task 2: Bullet Point Rewriter ---

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
You are a professional resume coach helping a career changer.
Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
Use strong action verbs. Do not invent facts that aren't implied by the original.

Return ONLY a valid JSON list.
Do NOT include markdown, code fences, or explanations.
Do NOT wrap the JSON in ```json blocks.

Each item should have two keys:
"original" (the original bullet)
"improved" (your rewritten version)

Bullet points:

"""

    messages = [{"role": "user", "content": prompt}]
    raw_response = get_completion(messages, temperature=0.4)

    try:
        results = json.loads(raw_response)
    except json.JSONDecodeError:
        print("\nJSON parsing failed. Raw response:")
        print(raw_response)
        return []

    print("\nRewritten Resume Bullets:")
    for item in results:
        print("\nOriginal:", item["original"])
        print("Improved:", item["improved"])

    return results


# These starter bullets are weak because they are vague and do not show results, tools, skills, or measurable impact.
# The model should suggest stronger action verbs, clearer responsibilities, and more professional wording.


# --- Task 3: Cover Letter Generator ---

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
You write strong cover letter opening paragraphs for career changers.
The paragraph should be 3-5 sentences: confident, specific, and free of clichés.
Do not invent experience, credentials, or achievements that are not provided.

Here are two examples of the style and tone you should match:

Example 1:
Role: Data Analyst at a healthcare nonprofit
Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
Opening: After seven years as a registered nurse, I've spent my career making decisions
under pressure using incomplete information — which turns out to be excellent training for
data analysis. I recently completed a data analytics program where I built dashboards
tracking patient outcomes across departments. I'm excited to bring that combination of
clinical context and technical skill to [Company]'s mission-driven work.

Example 2:
Role: Junior Software Engineer at a fintech startup
Background: Ten years in retail banking operations, self-taught Python developer for two years.
Opening: I spent a decade on the operations side of banking, watching technology decisions
get made by people who had never processed a wire transfer or resolved a failed ACH batch.
That frustration turned into curiosity, and two years of self-teaching Python later, I'm
ready to be on the other side of those decisions. I'm applying to [Company] because your
work on payment infrastructure is exactly where my domain expertise and new technical skills
intersect.

Now write an opening paragraph for this person:
Role: {job_title}
Background: {background}
Opening:
"""

    messages = [{"role": "user", "content": prompt}]
    return get_completion(messages, temperature=0.7)


# I chose these examples because they show how a career changer can connect past experience to a new technical role.
# The few-shot pattern helps control the tone, structure, length, and level of detail in the output.


# --- Task 4: Moderation Check ---

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )

    flagged = result.results[0].flagged

    if flagged:
        print("\nJob Application Helper: I cannot help with that wording. Please rephrase your request in a safer and more respectful way.")
        return False

    return True


# --- Task 5: The Chatbot Loop ---

def run_chatbot():
    messages = [
        {"role": "system", "content": YOUR_SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        if not user_input:
            continue

        if not is_safe(user_input):
            continue

        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")

            raw_bullets = []

            while True:
                line = input().strip()

                if line.upper() == "DONE":
                    break

                if line:
                    if is_safe(line):
                        raw_bullets.append(line)

            if raw_bullets:
                rewritten = rewrite_bullets(raw_bullets)

                if rewritten:
                    print("\nJob Application Helper: Please review and edit these before submitting them anywhere.")
            else:
                print("\nJob Application Helper: No bullet points were entered.")

        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()

            if not is_safe(job_title) or not is_safe(background):
                continue

            paragraph = generate_cover_letter(job_title, background)

            print("\nCover Letter Opening:")
            print(paragraph)
            print("\nJob Application Helper: Please review and edit this before submitting it anywhere. I may not know every industry norm, so use your own judgment.")

        else:
            messages.append({"role": "user", "content": user_input})

            reply = get_completion(messages)

            print("\nJob Application Helper:")
            print(reply)
            print("\nReminder: Please review and edit any application material before submitting it anywhere.")

            messages.append({"role": "assistant", "content": reply})


# --- Test Calls ---

def run_tests():
    print("\n" + "=" * 50)
    print("Running Project 05 Test Calls")
    print("=" * 50)

    bullets = [
        "Helped customers with their problems",
        "Made reports for the management team",
        "Worked with a team to finish the project on time"
    ]

    rewrite_bullets(bullets)

    job_title = "Junior Data Engineer"
    background = "Five years of experience as a middle school math teacher; recently completed a Python course and built data pipelines using Prefect and Pandas."

    print("\nGenerated Cover Letter Opening:")
    print(generate_cover_letter(job_title, background))

    safe_input = "Can you help me rewrite my resume bullet points?"
    unsafe_input = "I want to hurt someone at work."

    print("\nModeration Test - Safe Input:")
    print(is_safe(safe_input))

    print("\nModeration Test - Flagged Input:")
    print(is_safe(unsafe_input))


if __name__ == "__main__":
    run_tests()
    run_chatbot()


# --- Task 6: Ethics Reflection ---
# Format chosen: Option A - Comment block
#
# A job application bot could produce biased advice because it was trained on text that may favor certain industries,
# writing styles, cultural backgrounds, or professional norms. For example, it might make a candidate sound overly formal
# or corporate even if that tone does not fit the company or role.
#
# If a job seeker submitted the bot's output without reviewing it, the application could include inaccurate details,
# exaggerated claims, or language that does not sound like the actual person. This could hurt the applicant if an employer
# asks follow-up questions and the applicant cannot honestly support what was written.
#
# One guardrail I would add professionally is a required review warning before copying or exporting any final answer.
# I would also include reminders not to invent experience and to verify that every claim is accurate.