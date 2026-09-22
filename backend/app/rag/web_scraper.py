from typing import Optional

import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import json
import os
import re
import time
from dotenv import load_dotenv

load_dotenv()

# Configure Groq (uses OpenAI-compatible API)
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


# ─── STEP 1: Whitelist of trusted article URLs ───
# These URLs serve static HTML content that can be scraped reliably.
# JS-rendered sites (OnePeloton, Gymshark) and blocker sites (Runner's World) are excluded.
TRUSTED_URLS = [
    # ── NUTRITION ──
    {"url": "https://www.nike.com/a/what-to-eat-before-running", "source": "Nike"},
    {"url": "https://www.nike.com/a/foods-for-runners", "source": "Nike"},
    {"url": "https://www.nike.com/a/running-recovery-tips", "source": "Nike"},
    {"url": "https://www.medicalnewstoday.com/articles/best-food-for-runners", "source": "Medical News Today"},
    {"url": "https://www.healthline.com/nutrition/best-foods-for-runners", "source": "Healthline"},
    {"url": "https://www.healthline.com/nutrition/electrolytes", "source": "Healthline"},
    {"url": "https://www.webmd.com/fitness-exercise/what-to-eat-before-run", "source": "WebMD"},
    {"url": "https://www.marathonpune.com/nutrition-general-diet.html", "source": "Pune Marathon"},
    {"url": "https://www.manipalcigna.com/diet/athletes-diet", "source": "Manipal Cigna"},

    # ── MOBILITY / STRETCHING ──
    {"url": "https://www.nike.com/a/running-warm-up", "source": "Nike"},
    {"url": "https://www.nike.com/a/stretches-after-running", "source": "Nike"},
    {"url": "https://www.healthline.com/health/fitness-exercise/essential-runner-stretches", "source": "Healthline"},
    {"url": "https://www.medicalnewstoday.com/articles/stretches-for-runners", "source": "Medical News Today"},
    {"url": "https://www.webmd.com/fitness-exercise/ss/slideshow-stretches-for-runners", "source": "WebMD"},

    # ── TRAINING PLANS ──
    {"url": "https://www.nike.com/running/marathon-training-plan", "source": "Nike"},
    {"url": "https://www.nike.com/a/how-to-start-running", "source": "Nike"},
    {"url": "https://www.healthline.com/health/how-to-start-running", "source": "Healthline"},
    {"url": "https://www.medicalnewstoday.com/articles/how-to-start-running", "source": "Medical News Today"},
    {"url": "https://www.8020endurance.com/intensity-guidelines-for-80-20-running/", "source": "80/20 Endurance"},
    {"url": "https://www.webmd.com/fitness-exercise/running-injuries-causes-prevention-treatment", "source": "WebMD"},

    # ── MEDICAL / PREVENTIVE HEALTH ──
    {"url": "https://www.healthline.com/health/iron-deficiency-in-athletes", "source": "Healthline"},
    {"url": "https://www.healthline.com/nutrition/vitamin-d-for-athletes", "source": "Healthline"},
    {"url": "https://www.medicalnewstoday.com/articles/runners-knee", "source": "Medical News Today"},
    {"url": "https://www.webmd.com/fitness-exercise/shin-splints", "source": "WebMD"},
    {"url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10302780/", "source": "PMC/NIH"},
    {"url": "https://www.coreorthosports.com/a-runners-guide-to-symptoms-treatment-and-prevention-of-common-running-injuries/", "source": "Core Ortho Sports"},
]

# ─── STEP 2 & 3: Scrape and clean a webpage ───
def scrape_article(url: str) -> Optional[str]:
    """Fetch a URL and extract the main text content."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted elements (ads, navs, scripts, footers)
        for tag in soup.find_all(["script", "style", "nav", "footer", "header",
                                   "aside", "form", "iframe", "noscript"]):
            tag.decompose()

        # Extract text from the article body
        # Try common article containers first
        article = soup.find("article") or soup.find("main") or soup.find("body")

        if article is None:
            print(f"  ⚠️  Could not find content in {url}")
            return None

        # Get clean text
        text = article.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        # Skip if too short (probably a paywall or failed scrape)
        if len(text) < 200:
            print(f"  ⚠️  Content too short ({len(text)} chars), skipping {url}")
            return None

        # Truncate if extremely long (keep first ~5000 chars for quality)
        if len(text) > 5000:
            text = text[:5000]

        print(f"  ✅ Scraped {len(text)} characters")
        return text

    except requests.exceptions.HTTPError as e:
        print(f"  ❌ BLOCKED by website (HTTP {e.response.status_code}): {url}")
        print(f"     → This site blocks scrapers. Skipping.")
        return None
    except requests.exceptions.Timeout:
        print(f"  ❌ TIMEOUT: {url} took too long to respond. Skipping.")
        return None
    except Exception as e:
        print(f"  ❌ Error scraping {url}: {e}")
        return None


# ─── STEP 4: Use Groq (Llama 3) to auto-generate metadata (with retry) ───
def generate_metadata(text: str, source: str) -> dict:
    """Use Groq's Llama 3 to automatically classify and tag the article."""
    prompt = f"""Analyze the following article about running and return a JSON object 
with these exact fields:

- "category": one of ["nutrition", "mobility", "training", "medical"]
- "distance_tier": one of ["5k", "10k", "half_marathon", "marathon", "all"]  
- "experience_level": one of ["beginner", "intermediate", "advanced", "all"]
- "topic": a short 2-4 word topic label (e.g., "pre_race_fueling", "dynamic_warmup")
- "title": a clean title for this article

Return ONLY valid JSON. No markdown, no explanation.

Article text:
{text[:3000]}
"""
    # Retry up to 3 times (handles rate limiting)
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
            )
            # Parse JSON from response
            json_str = (response.choices[0].message.content or "").strip()
            # Remove markdown code fences if present
            json_str = json_str.replace("```json", "").replace("```", "").strip()
            metadata = json.loads(json_str)
            metadata["source"] = source
            print(f"  🏷️  Tagged as: {metadata.get('category')} / {metadata.get('topic')}")
            return metadata
        except Exception as e:
            if attempt < 2:
                wait_time = (attempt + 1) * 5  # Wait 5s, then 10s
                print(f"  ⏳ Groq rate limited. Retrying in {wait_time}s... (attempt {attempt+1}/3)")
                time.sleep(wait_time)
            else:
                print(f"  ❌ Groq failed after 3 attempts: {e}")
                return {
                    "category": "unknown",
                    "distance_tier": "all",
                    "experience_level": "all",
                    "topic": "general",
                    "title": "Untitled Article",
                    "source": source
                }

    return {
        "category": "unknown",
        "distance_tier": "all",
        "experience_level": "all",
        "topic": "general",
        "title": "Untitled Article",
        "source": source
    }


# ─── STEP 5: Save as .md file with frontmatter ───
def save_as_markdown(text: str, metadata: dict, output_dir: str, index: int):
    """Save the scraped content as a clean .md file with YAML frontmatter."""
    # Create a safe filename from the topic
    safe_name = re.sub(r"[^a-z0-9_]", "_", metadata.get("topic", "article").lower())
    filename = f"{safe_name}_{index}.md"
    filepath = os.path.join(output_dir, filename)

    frontmatter = f"""---
category: {metadata.get('category', 'unknown')}
distance_tier: {metadata.get('distance_tier', 'all')}
experience_level: {metadata.get('experience_level', 'all')}
topic: {metadata.get('topic', 'general')}
source: "{metadata.get('source', 'unknown')}"
---"""

    title = metadata.get("title", "Untitled")
    content = f"{frontmatter}\n\n# {title}\n\n{text}"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  💾 Saved: {filename}")
    return filepath


# ─── MAIN PIPELINE ───
def run_ingestion_pipeline():
    """Run the full automated ingestion pipeline."""
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("🚀 AUTOMATED INGESTION PIPELINE")
    print("=" * 60)
    print(f"📋 Processing {len(TRUSTED_URLS)} URLs...\n")

    saved_files = []
    failed_scrapes = []
    failed_tags = []

    for i, entry in enumerate(TRUSTED_URLS):
        url = entry["url"]
        source = entry.get("source", "unknown")

        print(f"\n[{i+1}/{len(TRUSTED_URLS)}] 🌐 {source}")
        print(f"    URL: {url}")

        # Step 2-3: Scrape and clean
        text = scrape_article(url)
        if text is None:
            failed_scrapes.append(source)
            continue

        # Rate limit: wait 4 seconds between Gemini calls (free tier = 15 req/min)
        time.sleep(4)

        # Step 4: Auto-tag with Gemini
        metadata = generate_metadata(text, source)
        if metadata.get("category") == "unknown":
            failed_tags.append(source)

        # Step 5: Save as .md
        filepath = save_as_markdown(text, metadata, output_dir, i)
        saved_files.append(filepath)

    # ─── SUMMARY REPORT ───
    print("\n" + "=" * 60)
    print(f"✅ PIPELINE COMPLETE")
    print(f"   📄 Saved: {len(saved_files)}/{len(TRUSTED_URLS)} articles")
    print(f"   ❌ Failed to scrape: {len(failed_scrapes)} ({', '.join(failed_scrapes)})")
    print(f"   ⚠️  Failed to tag: {len(failed_tags)} ({', '.join(failed_tags)})")
    print(f"📂 Output: {output_dir}")
    print("=" * 60)



if __name__ == "__main__":
    run_ingestion_pipeline()

