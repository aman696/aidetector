"""
Configuration for social media screenshot scraper.

INSTRUCTIONS:
  1. Review and edit the account lists below before running
  2. AI accounts must EXCLUSIVELY post AI-generated images
  3. Real accounts must EXCLUSIVELY post real photographs
  4. Verify accounts still exist before scraping
  5. Add/remove accounts as needed — more accounts = bigger dataset

The defaults below are well-known, verified sources.
Reddit subreddits are the most reliable (clear community rules).
Instagram/X accounts should be double-checked before use.
"""

# ── Accounts that post AI-generated images ──────────────────────────────

AI_ACCOUNTS = {
    "instagram": [
    # AI art showcase accounts — active as of 2025/2026
    "midjourneyartwork",       # Midjourney community showcase (large, active)
    "lilmiquela",           
    "aidesign.png",             # Elmo Mistiaen — AI fashion, heavily featured
    "aiartdaily",               # General AI art reposts — still active
    "imma.gram",
    "millasofiafin",
    "kyraonig",
    "shudu.gram",
    "rozy.gram",
    "bermudaisbae",
    "fit_aitana"
],
"x": [
    # X accounts posting AI-generated images regularly
    "craftian_keskin",          # Keskin — hyper-realistic AI photo prompts (very active 2025)
    "SimplyAnnisa",             # Moody ultra-realistic AI portraits
    "astronomerozge1",          # Astrophysics + AI surreal art
    "LearnWithAbbay",
    "egeberkina",
    "generated_media",
    "YaseenK7212"
    ],
"reddit": [
    # Subreddits with exclusively or primarily AI-generated content
    "r/midjourney",             # Midjourney creations (all AI)
    "r/StableDiffusion",        # Stable Diffusion art (all AI)
    "r/aiArt",                  # General AI art
    "r/dalle2",                 # DALL-E creations
    "r/FluxAI",                 # Flux (Black Forest Labs) — newer, very active 2025
    "r/aiArt",            # AI wallpapers — exclusively generated
    "r/generative",  # Same niche, different community
],
}

# ── Accounts that post real photographs ─────────────────────────────────

REAL_ACCOUNTS = {
    "instagram": [
        "natgeo",                   # National Geographic (real photography)
        "stevemccurryofficial",     # Steve McCurry (photojournalist)
        "jimmychin",                # Jimmy Chin (adventure photographer)
        "paulnicklen",              # Paul Nicklen (wildlife photographer)
    ],
    "x": [
        "NatGeo",                   # National Geographic
        "Reuters",                  # Reuters news photography
        "AP",                       # Associated Press photography
    ],
    "reddit": [
        # ✓ These subreddits require REAL original photos (strictly moderated)
        "r/itookapicture",          # OC photography only
        "r/EarthPorn",              # Real nature photography
        "r/analog",                 # Film photography (physically cannot be AI)
        "r/streetphotography",      # Real street photography
    ],
   
}

# ── Scraper settings ────────────────────────────────────────────────────

SCRAPER_SETTINGS = {
    # Path to your Firefox profile (has login cookies)
    # Found via: cat ~/.mozilla/firefox/profiles.ini
    "firefox_profile": "~/.mozilla/firefox/ddzbh5kh.default-esr",

    # Max screenshots per account/subreddit
    "max_screenshots_per_account": 15,

    # Page load timeout (seconds)
    "page_load_timeout_sec": 20,

    # Delay between actions (seconds) — don't go below 1.0 to avoid bans
    "min_delay_sec": 1.5,
    "max_delay_sec": 3.0,
}
