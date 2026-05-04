import os
import re
import json
import smtplib
from urllib.parse import quote
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
import streamlit as st
import pdfplumber

from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine, text
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# 1. CONFIGURATION
# =====================================================

load_dotenv()


def get_secret(key):
    value = os.getenv(key)
    if value:
        return value

    try:
        return st.secrets[key]
    except Exception:
        return None


OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
DATABASE_URL = get_secret("DATABASE_URL")
ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD")
SMTP_EMAIL = get_secret("SMTP_EMAIL")
SMTP_PASSWORD = get_secret("SMTP_PASSWORD")

st.set_page_config(
    page_title="Abdou AI Recrutement",
    page_icon="🤖",
    layout="wide"
)

if not OPENAI_API_KEY:
    st.error("Clé OpenAI non trouvée")
    st.stop()

if not DATABASE_URL:
    st.error("DATABASE_URL non trouvée")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
engine = create_engine(DATABASE_URL)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# =====================================================
# 2. BRANDING
# =====================================================

APP_NAME = "Abdou AI Recrutement"
COMPANY_NAME = "Abdou Data & IA"
COMPANY_SECTOR = "Data, Intelligence Artificielle et Recrutement"
COMPANY_LOCATION = "France"
COMPANY_WEBSITE = ""
CONTACT_EMAIL = "contact@abdou-ai.com"
FOUNDER_NAME = "Abdou SECK"


# =====================================================
# 3. UTILITAIRES
# =====================================================

def clean_text(value):
    if value is None:
        return ""
    return str(value).replace("\x00", "").strip()


def safe_display(value):
    value = clean_text(value)
    if value.lower() in ["none", "null", "nan", ""]:
        return ""
    return value


def extract_email(text_value):
    emails = re.findall(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        clean_text(text_value)
    )
    return emails[0] if emails else ""


def extract_text_from_pdf(uploaded_file):
    text_content = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_content += page_text + "\n"
    return clean_text(text_content)


def save_uploaded_file(uploaded_file):
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", uploaded_file.name)
    file_path = os.path.join(UPLOAD_DIR, safe_name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def send_confirmation_email(candidate_email, candidate_name, job_title):
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        return False

    subject = "Confirmation de réception de votre candidature"

    body = f"""Bonjour {candidate_name},

Nous vous confirmons la bonne réception de votre candidature pour le poste suivant :

{job_title}

Votre dossier a bien été enregistré dans notre plateforme de recrutement.

Notre équipe procédera à l’analyse de votre profil et reviendra vers vous si votre candidature correspond aux critères du poste.

Cordialement,
{FOUNDER_NAME}
{COMPANY_NAME}
"""

    try:
        message = MIMEMultipart()
        message["From"] = SMTP_EMAIL
        message["To"] = candidate_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain", "utf-8"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.send_message(message)

        return True

    except Exception as e:
        st.warning(f"Candidature enregistrée, mais email non envoyé : {e}")
        return False


def embed_text(text_value):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=clean_text(text_value)[:3000]
    )
    return response.data[0].embedding


def compute_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]


# =====================================================
# 4. BASE DE DONNÉES
# =====================================================

def create_tables():
    with engine.begin() as conn:

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS companies (
                id SERIAL PRIMARY KEY,
                name TEXT,
                sector TEXT,
                location TEXT,
                website TEXT,
                description TEXT,
                values_text TEXT,
                contact_email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS job_offers (
                id SERIAL PRIMARY KEY,
                company_id INTEGER REFERENCES companies(id),
                title TEXT,
                company_name TEXT,
                location TEXT,
                contract_type TEXT,
                description TEXT,
                requirements TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS cvs (
                id SERIAL PRIMARY KEY,
                candidate_name TEXT,
                candidate_email TEXT,
                phone TEXT,
                linkedin TEXT,
                availability TEXT,
                salary_expectation TEXT,
                experience_level TEXT,
                desired_contract TEXT,
                job_title TEXT,
                motivation TEXT,
                file_name TEXT,
                file_path TEXT,
                content TEXT,
                file_data BYTEA,
                job_offer_id INTEGER REFERENCES job_offers(id),
                status TEXT DEFAULT 'Candidature reçue',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS cv_scores (
                id SERIAL PRIMARY KEY,
                job_offer_id INTEGER REFERENCES job_offers(id),
                cv_id INTEGER REFERENCES cvs(id),
                score INTEGER,
                similarity FLOAT,
                final_score INTEGER,
                technical_skills_score INTEGER,
                experience_score INTEGER,
                education_score INTEGER,
                tools_score INTEGER,
                global_fit_score INTEGER,
                summary TEXT,
                strengths TEXT,
                weaknesses TEXT,
                recommendation TEXT,
                interview_questions TEXT,
                explanation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

        migrations = [
            "ALTER TABLE job_offers ADD COLUMN IF NOT EXISTS company_id INTEGER REFERENCES companies(id);",
            "ALTER TABLE job_offers ADD COLUMN IF NOT EXISTS company_name TEXT;",
            "ALTER TABLE job_offers ADD COLUMN IF NOT EXISTS location TEXT;",
            "ALTER TABLE job_offers ADD COLUMN IF NOT EXISTS contract_type TEXT;",
            "ALTER TABLE job_offers ADD COLUMN IF NOT EXISTS requirements TEXT;",
            "ALTER TABLE job_offers ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;",
            "ALTER TABLE job_offers ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;",
            "ALTER TABLE cvs ADD COLUMN IF NOT EXISTS candidate_email TEXT;",
            "ALTER TABLE cvs ADD COLUMN IF NOT EXISTS phone TEXT;",
            "ALTER TABLE cvs ADD COLUMN IF NOT EXISTS linkedin TEXT;",
            "ALTER TABLE cvs ADD COLUMN IF NOT EXISTS availability TEXT;",
            "ALTER TABLE cvs ADD COLUMN IF NOT EXISTS salary_expectation TEXT;",
            "ALTER TABLE cvs ADD COLUMN IF NOT EXISTS experience_level TEXT;",
            "ALTER TABLE cvs ADD COLUMN IF NOT EXISTS desired_contract TEXT;",
            "ALTER TABLE cvs ADD COLUMN IF NOT EXISTS job_title TEXT;",
            "ALTER TABLE cvs ADD COLUMN IF NOT EXISTS motivation TEXT;",
            "ALTER TABLE cvs ADD COLUMN IF NOT EXISTS file_path TEXT;",
            "ALTER TABLE cvs ADD COLUMN IF NOT EXISTS file_data BYTEA;",
            "ALTER TABLE cvs ADD COLUMN IF NOT EXISTS job_offer_id INTEGER REFERENCES job_offers(id);",
            "ALTER TABLE cvs ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'Candidature reçue';",
            "ALTER TABLE cv_scores ADD COLUMN IF NOT EXISTS similarity FLOAT;",
            "ALTER TABLE cv_scores ADD COLUMN IF NOT EXISTS final_score INTEGER;",
            "ALTER TABLE cv_scores ADD COLUMN IF NOT EXISTS technical_skills_score INTEGER;",
            "ALTER TABLE cv_scores ADD COLUMN IF NOT EXISTS experience_score INTEGER;",
            "ALTER TABLE cv_scores ADD COLUMN IF NOT EXISTS education_score INTEGER;",
            "ALTER TABLE cv_scores ADD COLUMN IF NOT EXISTS tools_score INTEGER;",
            "ALTER TABLE cv_scores ADD COLUMN IF NOT EXISTS global_fit_score INTEGER;",
            "ALTER TABLE cv_scores ADD COLUMN IF NOT EXISTS summary TEXT;",
            "ALTER TABLE cv_scores ADD COLUMN IF NOT EXISTS strengths TEXT;",
            "ALTER TABLE cv_scores ADD COLUMN IF NOT EXISTS weaknesses TEXT;",
            "ALTER TABLE cv_scores ADD COLUMN IF NOT EXISTS recommendation TEXT;",
            "ALTER TABLE cv_scores ADD COLUMN IF NOT EXISTS interview_questions TEXT;",
            "ALTER TABLE cv_scores ADD COLUMN IF NOT EXISTS explanation TEXT;"
        ]

        for migration in migrations:
            conn.execute(text(migration))


def fix_candidate_email_constraints():
    """
    Correction importante :
    - Une candidature spontanée ne bloque plus une candidature à une offre.
    - Un candidat peut postuler à plusieurs offres différentes avec le même email.
    - Le doublon est bloqué uniquement pour la même offre.
    """
    with engine.begin() as conn:
        conn.execute(text("""
            ALTER TABLE cvs DROP CONSTRAINT IF EXISTS cvs_candidate_email_key;
        """))

        conn.execute(text("""
            DROP INDEX IF EXISTS unique_candidate_email;
        """))

        conn.execute(text("""
            DROP INDEX IF EXISTS unique_candidate_per_offer;
        """))

        conn.execute(text("""
            CREATE UNIQUE INDEX unique_candidate_per_offer
            ON cvs (LOWER(candidate_email), job_offer_id)
            WHERE job_offer_id IS NOT NULL;
        """))


def ensure_default_company():
    with engine.begin() as conn:
        company = conn.execute(text("""
            SELECT id FROM companies ORDER BY id ASC LIMIT 1
        """)).fetchone()

        if not company:
            conn.execute(text("""
                INSERT INTO companies (
                    name, sector, location, website, description, values_text, contact_email
                )
                VALUES (
                    :name, :sector, :location, :website, :description, :values_text, :contact_email
                )
            """), {
                "name": COMPANY_NAME,
                "sector": COMPANY_SECTOR,
                "location": COMPANY_LOCATION,
                "website": COMPANY_WEBSITE,
                "description": f"{COMPANY_NAME} est une plateforme intelligente de recrutement permettant de publier des offres, recevoir des candidatures et analyser automatiquement les CV grâce à l’IA.",
                "values_text": "Innovation, automatisation, transparence, performance et recrutement data-driven.",
                "contact_email": CONTACT_EMAIL
            })


def clean_empty_offers():
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE job_offers
            SET is_active = FALSE
            WHERE title IS NULL
            OR TRIM(title) = ''
            OR LOWER(TRIM(title)) = 'none';
        """))


# =====================================================
# 5. ENTREPRISE / OFFRES / CANDIDATS
# =====================================================

def load_company():
    with engine.begin() as conn:
        return conn.execute(text("""
            SELECT id, name, sector, location, website, description, values_text, contact_email, created_at
            FROM companies
            ORDER BY id ASC
            LIMIT 1
        """)).fetchone()


def save_job_offer(company_id, title, company_name, location, contract_type, description, requirements):
    with engine.begin() as conn:
        result = conn.execute(text("""
            INSERT INTO job_offers (
                company_id, title, company_name, location, contract_type, description, requirements, is_active
            )
            VALUES (
                :company_id, :title, :company_name, :location, :contract_type, :description, :requirements, TRUE
            )
            RETURNING id
        """), {
            "company_id": company_id,
            "title": clean_text(title),
            "company_name": clean_text(company_name),
            "location": clean_text(location),
            "contract_type": clean_text(contract_type),
            "description": clean_text(description),
            "requirements": clean_text(requirements)
        })
        return result.scalar()


def update_job_offer(offer_id, title, location, contract_type, description, requirements, is_active):
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE job_offers
            SET title = :title,
                location = :location,
                contract_type = :contract_type,
                description = :description,
                requirements = :requirements,
                is_active = :is_active,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :offer_id
        """), {
            "offer_id": offer_id,
            "title": clean_text(title),
            "location": clean_text(location),
            "contract_type": clean_text(contract_type),
            "description": clean_text(description),
            "requirements": clean_text(requirements),
            "is_active": is_active
        })


def deactivate_job_offer(offer_id):
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE job_offers
            SET is_active = FALSE,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :offer_id
        """), {"offer_id": offer_id})


def reactivate_job_offer(offer_id):
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE job_offers
            SET is_active = TRUE,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = :offer_id
        """), {"offer_id": offer_id})


def delete_job_offer(offer_id):
    with engine.begin() as conn:
        candidates_count = conn.execute(text("""
            SELECT COUNT(*)
            FROM cvs
            WHERE job_offer_id = :offer_id
        """), {"offer_id": offer_id}).scalar()

        if candidates_count > 0:
            conn.execute(text("""
                UPDATE job_offers
                SET is_active = FALSE,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = :offer_id
            """), {"offer_id": offer_id})
            return False

        conn.execute(text("""
            DELETE FROM job_offers
            WHERE id = :offer_id
        """), {"offer_id": offer_id})
        return True


def load_active_offers():
    company = load_company()
    if not company:
        return []

    with engine.begin() as conn:
        return conn.execute(text("""
            SELECT id, title, company_name, location, contract_type, description, requirements, created_at
            FROM job_offers
            WHERE is_active = TRUE
            AND company_id = :company_id
            AND title IS NOT NULL
            AND TRIM(title) <> ''
            ORDER BY created_at DESC
        """), {"company_id": company[0]}).fetchall()


def load_all_offers():
    company = load_company()
    if not company:
        return []

    with engine.begin() as conn:
        return conn.execute(text("""
            SELECT id, title, company_name, location, contract_type, description, requirements, is_active, created_at, updated_at
            FROM job_offers
            WHERE company_id = :company_id
            AND title IS NOT NULL
            AND TRIM(title) <> ''
            ORDER BY created_at DESC
        """), {"company_id": company[0]}).fetchall()


def load_offer_by_id(offer_id):
    with engine.begin() as conn:
        return conn.execute(text("""
            SELECT id, title, company_name, location, contract_type, description, requirements, created_at
            FROM job_offers
            WHERE id = :offer_id
        """), {"offer_id": offer_id}).fetchone()


def save_cv(candidate_name, candidate_email, phone, linkedin, availability,
            salary_expectation, experience_level, desired_contract,
            file_name, file_path, content, file_data=None, job_offer_id=None,
            job_title="", motivation=""):
    with engine.begin() as conn:
        result = conn.execute(text("""
            INSERT INTO cvs (
                candidate_name, candidate_email, phone, linkedin, availability,
                salary_expectation, experience_level, desired_contract,
                job_title, motivation, file_name, file_path, content, file_data,
                job_offer_id, status
            )
            VALUES (
                :candidate_name, :candidate_email, :phone, :linkedin, :availability,
                :salary_expectation, :experience_level, :desired_contract,
                :job_title, :motivation, :file_name, :file_path, :content, :file_data,
                :job_offer_id, 'Candidature reçue'
            )
            RETURNING id
        """), {
            "candidate_name": clean_text(candidate_name),
            "candidate_email": clean_text(candidate_email),
            "phone": clean_text(phone),
            "linkedin": clean_text(linkedin),
            "availability": clean_text(availability),
            "salary_expectation": clean_text(salary_expectation),
            "experience_level": clean_text(experience_level),
            "desired_contract": clean_text(desired_contract),
            "job_title": clean_text(job_title),
            "motivation": clean_text(motivation),
            "file_name": clean_text(file_name),
            "file_path": clean_text(file_path),
            "content": clean_text(content),
            "file_data": file_data,
            "job_offer_id": job_offer_id
        })
        return result.scalar()


def candidature_exists(candidate_email, job_offer_id):
    with engine.begin() as conn:
        count = conn.execute(text("""
            SELECT COUNT(*)
            FROM cvs
            WHERE LOWER(candidate_email) = LOWER(:candidate_email)
            AND job_offer_id = :job_offer_id
        """), {
            "candidate_email": clean_text(candidate_email),
            "job_offer_id": job_offer_id
        }).scalar()
        return count > 0


def load_candidates_by_offer(job_offer_id):
    with engine.begin() as conn:
        return conn.execute(text("""
            SELECT id, candidate_name, candidate_email, phone, linkedin,
                   availability, salary_expectation, experience_level,
                   desired_contract, job_title, motivation, file_name,
                   file_path, content, file_data, status, created_at
            FROM cvs
            WHERE job_offer_id = :job_offer_id
            ORDER BY created_at DESC
        """), {"job_offer_id": job_offer_id}).fetchall()


def load_spontaneous_candidates():
    with engine.begin() as conn:
        return conn.execute(text("""
            SELECT id, candidate_name, candidate_email, phone, linkedin,
                   availability, salary_expectation, experience_level,
                   desired_contract, job_title, motivation, file_name,
                   file_path, content, file_data, status, created_at
            FROM cvs
            WHERE job_offer_id IS NULL
            ORDER BY created_at DESC
        """)).fetchall()


def load_all_candidates():
    with engine.begin() as conn:
        return conn.execute(text("""
            SELECT id, candidate_name, candidate_email, phone, linkedin,
                   availability, salary_expectation, experience_level,
                   desired_contract, job_title, motivation, file_name,
                   file_path, content, file_data, status, job_offer_id, created_at
            FROM cvs
            ORDER BY created_at DESC
        """)).fetchall()


def update_candidate_status(cv_id, status):
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE cvs SET status = :status WHERE id = :cv_id
        """), {"status": status, "cv_id": cv_id})


def load_dashboard_stats():
    with engine.begin() as conn:
        offers_count = conn.execute(text("""
            SELECT COUNT(*) FROM job_offers WHERE is_active = TRUE
        """)).scalar()
        candidates_count = conn.execute(text("SELECT COUNT(*) FROM cvs")).scalar()
        spontaneous_count = conn.execute(text("SELECT COUNT(*) FROM cvs WHERE job_offer_id IS NULL")).scalar()
        best_score = conn.execute(text("SELECT MAX(final_score) FROM cv_scores")).scalar()
        avg_score = conn.execute(text("SELECT AVG(final_score) FROM cv_scores")).scalar()

    return {
        "offers_count": offers_count or 0,
        "candidates_count": candidates_count or 0,
        "spontaneous_count": spontaneous_count or 0,
        "best_score": int(best_score) if best_score else 0,
        "avg_score": int(avg_score) if avg_score else 0
    }


def save_score(job_offer_id, cv_id, result):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO cv_scores (
                job_offer_id, cv_id, score, similarity, final_score,
                technical_skills_score, experience_score, education_score,
                tools_score, global_fit_score, summary, strengths,
                weaknesses, recommendation, interview_questions, explanation
            )
            VALUES (
                :job_offer_id, :cv_id, :score, :similarity, :final_score,
                :technical_skills_score, :experience_score, :education_score,
                :tools_score, :global_fit_score, :summary, :strengths,
                :weaknesses, :recommendation, :interview_questions, :explanation
            )
        """), {
            "job_offer_id": job_offer_id,
            "cv_id": cv_id,
            "score": int(result.get("score", 0)),
            "similarity": float(result.get("similarity", 0)),
            "final_score": int(result.get("final_score", 0)),
            "technical_skills_score": int(result.get("technical_skills_score", 0)),
            "experience_score": int(result.get("experience_score", 0)),
            "education_score": int(result.get("education_score", 0)),
            "tools_score": int(result.get("tools_score", 0)),
            "global_fit_score": int(result.get("global_fit_score", 0)),
            "summary": clean_text(result.get("summary", "")),
            "strengths": clean_text(result.get("strengths", "")),
            "weaknesses": clean_text(result.get("weaknesses", "")),
            "recommendation": clean_text(result.get("recommendation", "")),
            "interview_questions": clean_text(result.get("interview_questions", "")),
            "explanation": clean_text(result.get("explanation", ""))
        })


# =====================================================
# 6. ANALYSE IA
# =====================================================

def score_cv(job_offer_text, cv_text, file_name):
    detected_email = extract_email(cv_text)

    prompt = f"""
Tu es un expert RH spécialisé dans le recrutement.

Analyse le CV par rapport à l'offre d'emploi.

Donne une note sur 100 selon ces critères :
- Compétences techniques : 30 points
- Expérience professionnelle : 25 points
- Formation : 15 points
- Outils et technologies : 15 points
- Adéquation globale au poste : 15 points

Le score global doit être la somme des 5 critères et ne doit pas dépasser 100.

Réponds uniquement en JSON valide avec ce format :

{{
  "candidate_name": "Nom du candidat si détecté sinon Inconnu",
  "candidate_email": "{detected_email}",
  "file_name": "{file_name}",
  "score": 0,
  "technical_skills_score": 0,
  "experience_score": 0,
  "education_score": 0,
  "tools_score": 0,
  "global_fit_score": 0,
  "summary": "résumé court du profil",
  "strengths": "points forts du candidat",
  "weaknesses": "points faibles du candidat",
  "recommendation": "recommandation finale",
  "interview_questions": "5 questions d'entretien pertinentes",
  "explanation": "explication courte de la note"
}}

OFFRE D'EMPLOI :
{job_offer_text}

CV :
{cv_text[:10000]}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Tu es un recruteur expert. Tu réponds uniquement en JSON valide."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    result_text = response.choices[0].message.content

    try:
        result = json.loads(result_text)
    except json.JSONDecodeError:
        result = {
            "candidate_name": "Inconnu",
            "candidate_email": detected_email,
            "file_name": file_name,
            "score": 0,
            "technical_skills_score": 0,
            "experience_score": 0,
            "education_score": 0,
            "tools_score": 0,
            "global_fit_score": 0,
            "summary": "",
            "strengths": "",
            "weaknesses": "",
            "recommendation": "",
            "interview_questions": "",
            "explanation": result_text
        }

    for key in [
        "score", "technical_skills_score", "experience_score",
        "education_score", "tools_score", "global_fit_score"
    ]:
        try:
            result[key] = int(result.get(key, 0))
        except Exception:
            result[key] = 0

    return result


def analyze_candidate(job_offer_text, cv_text, file_name):
    result = score_cv(job_offer_text, cv_text, file_name)

    try:
        job_embedding = embed_text(job_offer_text)
        cv_embedding = embed_text(cv_text)
        similarity = compute_similarity(job_embedding, cv_embedding)
        result["similarity"] = round(similarity * 100, 2)
    except Exception:
        result["similarity"] = 0

    result["final_score"] = int(
        0.7 * int(result.get("score", 0)) +
        0.3 * float(result.get("similarity", 0))
    )

    return result


# =====================================================
# 7. INITIALISATION
# =====================================================

try:
    create_tables()
    fix_candidate_email_constraints()
    ensure_default_company()
    clean_empty_offers()
except Exception as e:
    st.error(f"Erreur PostgreSQL : {e}")
    st.stop()

if "application_offer_id" not in st.session_state:
    st.session_state.application_offer_id = None

if "df_results" not in st.session_state:
    st.session_state.df_results = None

if "selected_result_offer_id" not in st.session_state:
    st.session_state.selected_result_offer_id = None


# =====================================================
# 8. DESIGN CSS
# =====================================================

st.markdown("""
<style>
[data-testid="stSidebar"] {display: none;}
[data-testid="collapsedControl"] {display: none;}

.top-header {
    padding: 18px 25px;
    border-radius: 25px;
    background: white;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.06);
    margin-bottom: 25px;
}

.hero {
    background: linear-gradient(135deg, #0047bb, #00a7d8);
    padding: 55px;
    border-radius: 32px;
    margin-bottom: 30px;
    color: white;
}

.hero h1 {
    font-size: 46px;
    font-weight: 900;
    margin-bottom: 16px;
}

.hero p {
    font-size: 20px;
    max-width: 950px;
    line-height: 1.6;
}

.main-title {
    font-size: 34px;
    font-weight: 900;
    color: #0f172a;
    margin: 20px 0px;
}

.card, .job-card {
    padding: 26px;
    border-radius: 24px;
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    box-shadow: 0 6px 22px rgba(0,0,0,0.07);
    margin-bottom: 22px;
}

.job-card h3 {
    color: #0f4cc9;
    font-size: 26px;
    font-weight: 800;
}

.badge {
    background: #eaf2ff;
    color: #0057d9;
    padding: 8px 15px;
    border-radius: 999px;
    font-weight: 700;
    display: inline-block;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 18px;
    background-color: #ffffff;
    padding: 12px 18px;
    border-radius: 22px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.06);
    margin-bottom: 30px;
}

.stTabs [data-baseweb="tab"] {
    height: 55px;
    padding: 12px 28px;
    border-radius: 18px;
    font-size: 18px;
    font-weight: 800;
    color: #0f172a;
}

.stTabs [aria-selected="true"] {
    background-color: #0f4cc9 !important;
    color: white !important;
}

.stTabs [data-baseweb="tab-highlight"] {display: none;}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 9. HEADER
# =====================================================

st.markdown(f"""
<div class="top-header">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <div style="font-size:30px; font-weight:900; color:#0f4cc9;">{APP_NAME}</div>
            <div style="font-size:14px; color:#64748b;">Analyser • Recruter • Automatiser</div>
        </div>
        <div style="font-size:16px; color:#0f172a; font-weight:700;">
            Plateforme privée de recrutement IA
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 10. NAVIGATION PRINCIPALE
# =====================================================

tab_accueil, tab_candidat, tab_recruteur = st.tabs([
    "🏠 Accueil",
    "👤 Espace candidat",
    "🏢 Espace recruteur"
])


# =====================================================
# 11. ACCUEIL
# =====================================================

with tab_accueil:
    company = load_company()

    st.markdown(f"""
    <div class="hero">
        <h1>Bienvenue sur {APP_NAME}</h1>
        <p>
            La plateforme privée de {COMPANY_NAME} pour publier des offres,
            recevoir des candidatures et analyser automatiquement les CV grâce à l’intelligence artificielle.
        </p>
    </div>
    """, unsafe_allow_html=True)

    accueil_tab1, accueil_tab2, accueil_tab3, accueil_tab4 = st.tabs([
        "Présentation", "Nos valeurs", "Nos métiers", "Contact"
    ])

    with accueil_tab1:
        st.markdown('<div class="main-title">Présentation</div>', unsafe_allow_html=True)
        st.write(company[5] if company else f"{COMPANY_NAME} accompagne ses recrutements avec l’IA.")

    with accueil_tab2:
        st.markdown('<div class="main-title">Nos valeurs</div>', unsafe_allow_html=True)
        st.write(company[6] if company else "Innovation, transparence et performance.")

    with accueil_tab3:
        st.markdown('<div class="main-title">Nos offres</div>', unsafe_allow_html=True)
        offers = load_active_offers()
        if offers:
            for offer in offers:
                st.write(f"- **{safe_display(offer[1])}** — {safe_display(offer[3])} — {safe_display(offer[4])}")
        else:
            st.info("Aucune offre publiée pour le moment.")

    with accueil_tab4:
        st.markdown('<div class="main-title">Contact</div>', unsafe_allow_html=True)
        st.write(f"**Entreprise :** {COMPANY_NAME}")
        st.write(f"**Email :** {CONTACT_EMAIL}")
        st.write(f"**Localisation :** {COMPANY_LOCATION}")


# =====================================================
# 12. ESPACE CANDIDAT
# =====================================================

with tab_candidat:

    if st.session_state.application_offer_id:
        selected_offer = load_offer_by_id(st.session_state.application_offer_id)

        st.markdown(f"""
        <div class="hero">
            <h1>Finaliser votre candidature</h1>
            <p>Complétez vos informations et ajoutez votre CV pour postuler chez {COMPANY_NAME}.</p>
        </div>
        """, unsafe_allow_html=True)

        if not selected_offer:
            st.error("Offre introuvable.")
            if st.button("← Retour aux offres"):
                st.session_state.application_offer_id = None
                st.rerun()

        else:
            offer_id, title, company_name, location, contract_type, description, requirements, created_at = selected_offer

            title = safe_display(title)
            company_name = safe_display(company_name)
            location = safe_display(location)
            contract_type = safe_display(contract_type)
            description = safe_display(description)
            requirements = safe_display(requirements)

            if st.button("← Retour aux offres"):
                st.session_state.application_offer_id = None
                st.rerun()

            st.markdown(f"""
            <div class="job-card">
                <h3>{title}</h3>
                <p><b>Entreprise :</b> {company_name}</p>
                <p><b>Lieu :</b> {location}</p>
                <p><b>Contrat :</b> <span class="badge">{contract_type}</span></p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="main-title">Formulaire de candidature</div>', unsafe_allow_html=True)

            col_a, col_b = st.columns(2)

            with col_a:
                candidate_name = st.text_input("Nom complet", key=f"name_{offer_id}")
                candidate_email = st.text_input("Email", key=f"email_{offer_id}")
                phone = st.text_input("Téléphone", key=f"phone_{offer_id}")
                linkedin = st.text_input("LinkedIn", key=f"linkedin_{offer_id}")

            with col_b:
                availability = st.selectbox(
                    "Disponibilité",
                    [
                        "Immédiate",
                        "Sous 1 semaine",
                        "Sous 2 semaines",
                        "Sous 1 mois",
                        "Sous 3 mois",
                        "À définir"
                    ],
                    key=f"availability_{offer_id}"
                )

                salary_choice = st.selectbox(
                    "Prétentions salariales",
                    [
                        "Sélectionner",
                        "Moins de 30k €",
                        "30k - 40k €",
                        "40k - 50k €",
                        "50k - 60k €",
                        "60k - 80k €",
                        "80k € et plus",
                        "À négocier",
                        "Autre"
                    ],
                    key=f"salary_choice_{offer_id}"
                )

                salary_custom = ""
                if salary_choice == "Autre":
                    salary_custom = st.text_input("Précisez votre salaire", key=f"salary_custom_{offer_id}")

                salary_expectation = salary_custom if salary_custom else salary_choice

                experience_level = st.selectbox(
                    "Niveau d’expérience",
                    ["Débutant", "Junior", "Confirmé", "Senior", "Expert"],
                    key=f"experience_{offer_id}"
                )

                desired_contract = st.selectbox(
                    "Contrat recherché",
                    ["CDI", "CDD", "Stage", "Alternance", "Freelance", "Intérim"],
                    key=f"contract_{offer_id}"
                )

            motivation = st.text_area("Message de motivation", key=f"motivation_{offer_id}")
            candidate_cv = st.file_uploader("Ajoutez votre CV PDF", type="pdf", key=f"cv_{offer_id}")

            if st.button("Envoyer ma candidature", key=f"send_{offer_id}"):
                if not candidate_name or not candidate_email or not candidate_cv:
                    st.error("Veuillez compléter le nom, l’email et ajouter un CV.")

                elif candidature_exists(candidate_email, offer_id):
                    st.error("Vous avez déjà candidaté à cette offre avec cet email.")

                else:
                    file_path = save_uploaded_file(candidate_cv)
                    candidate_cv.seek(0)
                    cv_text = extract_text_from_pdf(candidate_cv)

                    save_cv(
                        candidate_name=candidate_name,
                        candidate_email=candidate_email,
                        phone=phone,
                        linkedin=linkedin,
                        availability=availability,
                        salary_expectation=salary_expectation,
                        experience_level=experience_level,
                        desired_contract=desired_contract,
                        file_name=candidate_cv.name,
                        file_path=file_path,
                        content=cv_text,
                        file_data=candidate_cv.getvalue(),
                        job_offer_id=offer_id,
                        job_title=title,
                        motivation=motivation
                    )

                    send_confirmation_email(
                        candidate_email=candidate_email,
                        candidate_name=candidate_name,
                        job_title=title
                    )

                    st.success("Votre candidature a bien été envoyée 🎉")
                    st.session_state.application_offer_id = None
                    st.rerun()

    else:
        st.markdown(f"""
        <div class="hero">
            <h1>Espace candidat</h1>
            <p>Consultez les offres disponibles et envoyez votre candidature en quelques clics.</p>
        </div>
        """, unsafe_allow_html=True)

        candidat_tab1, candidat_tab2 = st.tabs([
            "Recherche d’emploi",
            "Candidature spontanée"
        ])

        with candidat_tab1:
            st.markdown('<div class="main-title">Rechercher une offre</div>', unsafe_allow_html=True)

            col_search1, col_search2, col_search3 = st.columns(3)

            with col_search1:
                search_job = st.text_input("🔎 Recherche", placeholder="Data Analyst, Python, CDI...")

            with col_search2:
                filter_location = st.text_input("📍 Lieu", placeholder="Paris, Lyon, Remote...")

            with col_search3:
                filter_contract = st.selectbox(
                    "Contrat",
                    ["Tous", "CDI", "CDD", "Stage", "Alternance", "Freelance", "Intérim"]
                )

            offers = load_active_offers()

            if search_job:
                search = search_job.lower()
                offers = [
                    offer for offer in offers
                    if search in str(offer[1]).lower()
                    or search in str(offer[2]).lower()
                    or search in str(offer[3]).lower()
                    or search in str(offer[4]).lower()
                    or search in str(offer[5]).lower()
                    or search in str(offer[6]).lower()
                ]

            if filter_location:
                offers = [
                    offer for offer in offers
                    if filter_location.lower() in str(offer[3]).lower()
                ]

            if filter_contract != "Tous":
                offers = [
                    offer for offer in offers
                    if filter_contract.lower() in str(offer[4]).lower()
                ]

            if not offers:
                st.info("Aucune offre disponible pour le moment.")
            else:
                for offer in offers:
                    offer_id, title, company_name, location, contract_type, description, requirements, created_at = offer

                    title = safe_display(title)
                    company_name = safe_display(company_name)
                    location = safe_display(location)
                    contract_type = safe_display(contract_type)
                    description = safe_display(description)
                    requirements = safe_display(requirements)

                    short_description = description[:450] + "..." if len(description) > 450 else description

                    st.markdown(f"""
                    <div class="job-card">
                        <h3>{title}</h3>
                        <p><b>Entreprise :</b> {company_name}</p>
                        <p><b>Lieu :</b> {location}</p>
                        <p><b>Contrat :</b> <span class="badge">{contract_type}</span></p>
                        <p>{short_description}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("Consulter l’offre"):
                        st.write("### Description")
                        st.write(description)
                        if requirements:
                            st.write("### Compétences requises")
                            st.write(requirements)

                    if st.button("Candidater", key=f"apply_{offer_id}"):
                        st.session_state.application_offer_id = offer_id
                        st.rerun()

        with candidat_tab2:
            st.markdown('<div class="main-title">Candidature spontanée</div>', unsafe_allow_html=True)

            with st.form("form_spontanee"):
                col_a, col_b = st.columns(2)

                with col_a:
                    spontaneous_name = st.text_input("Nom complet")
                    spontaneous_email = st.text_input("Email")
                    spontaneous_phone = st.text_input("Téléphone")
                    spontaneous_linkedin = st.text_input("LinkedIn")

                with col_b:
                    spontaneous_job = st.text_input("Poste recherché")
                    spontaneous_availability = st.selectbox(
                        "Disponibilité",
                        [
                            "Immédiate",
                            "Sous 1 semaine",
                            "Sous 2 semaines",
                            "Sous 1 mois",
                            "Sous 3 mois",
                            "À définir"
                        ],
                        key="spontaneous_availability"
                    )

                    spontaneous_salary_choice = st.selectbox(
                        "Prétentions salariales",
                        [
                            "Sélectionner",
                            "Moins de 30k €",
                            "30k - 40k €",
                            "40k - 50k €",
                            "50k - 60k €",
                            "60k - 80k €",
                            "80k € et plus",
                            "À négocier",
                            "Autre"
                        ],
                        key="spontaneous_salary_choice"
                    )

                    spontaneous_salary_custom = ""
                    if spontaneous_salary_choice == "Autre":
                        spontaneous_salary_custom = st.text_input(
                            "Précisez votre salaire",
                            key="spontaneous_salary_custom"
                        )

                    spontaneous_salary = spontaneous_salary_custom if spontaneous_salary_custom else spontaneous_salary_choice

                    spontaneous_experience = st.selectbox(
                        "Niveau d’expérience",
                        ["Débutant", "Junior", "Confirmé", "Senior", "Expert"]
                    )

                    spontaneous_contract = st.selectbox(
                        "Contrat recherché",
                        ["CDI", "CDD", "Stage", "Alternance", "Freelance", "Intérim"]
                    )

                spontaneous_motivation = st.text_area("Message de motivation")
                spontaneous_cv = st.file_uploader("Ajoutez votre CV PDF", type="pdf", key="spontaneous_cv")

                submit_spontaneous = st.form_submit_button("Envoyer ma candidature spontanée")

                if submit_spontaneous:
                    if not spontaneous_name or not spontaneous_email or not spontaneous_cv:
                        st.error("Veuillez compléter le nom, l’email et ajouter un CV.")
                    else:
                        file_path = save_uploaded_file(spontaneous_cv)
                        spontaneous_cv.seek(0)
                        cv_text = extract_text_from_pdf(spontaneous_cv)

                        save_cv(
                            candidate_name=spontaneous_name,
                            candidate_email=spontaneous_email,
                            phone=spontaneous_phone,
                            linkedin=spontaneous_linkedin,
                            availability=spontaneous_availability,
                            salary_expectation=spontaneous_salary,
                            experience_level=spontaneous_experience,
                            desired_contract=spontaneous_contract,
                            file_name=spontaneous_cv.name,
                            file_path=file_path,
                            content=cv_text,
                            file_data=spontaneous_cv.getvalue(),
                            job_offer_id=None,
                            job_title=spontaneous_job,
                            motivation=spontaneous_motivation
                        )

                        send_confirmation_email(
                            candidate_email=spontaneous_email,
                            candidate_name=spontaneous_name,
                            job_title=spontaneous_job or "Candidature spontanée"
                        )

                        st.success("Votre candidature spontanée a bien été envoyée 🎉")


# =====================================================
# 13. ESPACE RECRUTEUR
# =====================================================

with tab_recruteur:
    if "recruteur_auth" not in st.session_state:
        st.session_state.recruteur_auth = False

    if not st.session_state.recruteur_auth:
        st.markdown(f"""
        <div class="hero">
            <h1>Accès recruteur sécurisé</h1>
            <p>Veuillez entrer le mot de passe administrateur pour accéder à l’espace recruteur de {COMPANY_NAME}.</p>
        </div>
        """, unsafe_allow_html=True)

        password_input = st.text_input("Mot de passe recruteur", type="password")

        if st.button("Se connecter"):
            if ADMIN_PASSWORD and password_input == ADMIN_PASSWORD:
                st.session_state.recruteur_auth = True
                st.success("Connexion réussie.")
                st.rerun()
            else:
                st.error("Mot de passe incorrect.")

        st.info("Cet espace est réservé au recruteur.")

    else:
        st.markdown(f"""
        <div class="hero">
            <h1>Espace recruteur</h1>
            <p>
                Gérez les recrutements de {COMPANY_NAME} : publication des offres,
                suivi des candidatures, analyse automatique des CV et réponses professionnelles.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Déconnexion recruteur"):
            st.session_state.recruteur_auth = False
            st.rerun()

        stats = load_dashboard_stats()

        st.markdown('<div class="main-title">Pilotage du recrutement</div>', unsafe_allow_html=True)

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Offres actives", stats["offers_count"])
        kpi2.metric("Candidatures reçues", stats["candidates_count"])
        kpi3.metric("Spontanées", stats["spontaneous_count"])
        kpi4.metric("Meilleur score IA", f"{stats['best_score']}/100")

        recruteur_tab1, recruteur_tab2, recruteur_tab3, recruteur_tab4, recruteur_tab5 = st.tabs([
            "Publier une offre",
            "Gérer les offres",
            "Candidats par offre",
            "Candidatures spontanées",
            "Toutes les candidatures"
        ])

        with recruteur_tab1:
            st.markdown('<div class="main-title">Publier une offre d’emploi</div>', unsafe_allow_html=True)

            company = load_company()

            if not company:
                st.error("Entreprise introuvable.")
            else:
                st.info(f"Entreprise : {company[1]}")

                col1, col2 = st.columns(2)

                with col1:
                    title = st.text_input("Titre du poste")
                    location = st.text_input("Lieu", value=safe_display(company[3]))

                with col2:
                    contract_type = st.selectbox(
                        "Type de contrat",
                        ["CDI", "CDD", "Stage", "Alternance", "Freelance", "Intérim"]
                    )

                description = st.text_area("Description de l’offre", height=240)
                requirements = st.text_area("Compétences requises", height=170)

                if st.button("Publier l’offre"):
                    if not clean_text(title) or not clean_text(description):
                        st.error("Veuillez renseigner au minimum le titre et la description.")
                    else:
                        save_job_offer(
                            company_id=company[0],
                            title=title,
                            company_name=company[1],
                            location=location,
                            contract_type=contract_type,
                            description=description,
                            requirements=requirements
                        )
                        st.success("Offre publiée avec succès 🎉")
                        st.rerun()

        with recruteur_tab2:
            st.markdown('<div class="main-title">Gérer les offres d’emploi</div>', unsafe_allow_html=True)

            offers = load_all_offers()

            if not offers:
                st.info("Aucune offre disponible.")
            else:
                offer_options = {
                    f"#{offer[0]} - {safe_display(offer[1])} - {'Active' if offer[7] else 'Inactive'}": offer[0]
                    for offer in offers
                }

                selected_offer_label = st.selectbox(
                    "Sélectionnez une offre à modifier",
                    list(offer_options.keys())
                )

                selected_offer_id = offer_options[selected_offer_label]

                selected_offer = None
                for offer in offers:
                    if offer[0] == selected_offer_id:
                        selected_offer = offer
                        break

                if selected_offer:
                    offer_id = selected_offer[0]
                    old_title = safe_display(selected_offer[1])
                    old_company_name = safe_display(selected_offer[2])
                    old_location = safe_display(selected_offer[3])
                    old_contract_type = safe_display(selected_offer[4])
                    old_description = safe_display(selected_offer[5])
                    old_requirements = safe_display(selected_offer[6])
                    old_is_active = bool(selected_offer[7])

                    st.markdown(f"""
                    <div class="job-card">
                        <h3>{old_title}</h3>
                        <p><b>Entreprise :</b> {old_company_name}</p>
                        <p><b>Lieu :</b> {old_location}</p>
                        <p><b>Contrat :</b> <span class="badge">{old_contract_type}</span></p>
                        <p><b>Statut :</b> {'Active' if old_is_active else 'Inactive'}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.subheader("Modifier l’offre")

                    new_title = st.text_input(
                        "Titre du poste",
                        value=old_title,
                        key=f"edit_title_{offer_id}"
                    )

                    new_location = st.text_input(
                        "Lieu",
                        value=old_location,
                        key=f"edit_location_{offer_id}"
                    )

                    contract_list = ["CDI", "CDD", "Stage", "Alternance", "Freelance", "Intérim"]

                    default_index = (
                        contract_list.index(old_contract_type)
                        if old_contract_type in contract_list
                        else 0
                    )

                    new_contract_type = st.selectbox(
                        "Type de contrat",
                        contract_list,
                        index=default_index,
                        key=f"edit_contract_{offer_id}"
                    )

                    new_description = st.text_area(
                        "Description de l’offre",
                        value=old_description,
                        height=220,
                        key=f"edit_description_{offer_id}"
                    )

                    new_requirements = st.text_area(
                        "Compétences requises",
                        value=old_requirements,
                        height=160,
                        key=f"edit_requirements_{offer_id}"
                    )

                    new_is_active = st.checkbox(
                        "Offre active visible côté candidat",
                        value=old_is_active,
                        key=f"edit_active_{offer_id}"
                    )

                    col_edit1, col_edit2, col_edit3 = st.columns(3)

                    with col_edit1:
                        if st.button("Enregistrer les modifications", key=f"save_edit_{offer_id}"):
                            if not clean_text(new_title) or not clean_text(new_description):
                                st.error("Le titre et la description sont obligatoires.")
                            else:
                                update_job_offer(
                                    offer_id=offer_id,
                                    title=new_title,
                                    location=new_location,
                                    contract_type=new_contract_type,
                                    description=new_description,
                                    requirements=new_requirements,
                                    is_active=new_is_active
                                )
                                st.success("Offre modifiée avec succès.")
                                st.rerun()

                    with col_edit2:
                        if old_is_active:
                            if st.button("Retirer l’offre", key=f"deactivate_{offer_id}"):
                                deactivate_job_offer(offer_id)
                                st.warning("Offre retirée de l’espace candidat.")
                                st.rerun()
                        else:
                            if st.button("Réactiver l’offre", key=f"reactivate_{offer_id}"):
                                reactivate_job_offer(offer_id)
                                st.success("Offre réactivée.")
                                st.rerun()

                    with col_edit3:
                        if st.button("Supprimer définitivement", key=f"delete_{offer_id}"):
                            deleted = delete_job_offer(offer_id)

                            if deleted:
                                st.success("Offre supprimée définitivement.")
                            else:
                                st.warning(
                                    "Cette offre contient déjà des candidatures. "
                                    "Elle a donc été retirée au lieu d’être supprimée."
                                )

                            st.rerun()

        with recruteur_tab3:
            st.markdown('<div class="main-title">Candidats par offre & Analyse IA</div>', unsafe_allow_html=True)

            offers = load_all_offers()

            if not offers:
                st.info("Aucune offre publiée pour le moment.")
            else:
                offer_options = {
                    f"#{offer[0]} - {safe_display(offer[1])} - {safe_display(offer[2])} - {safe_display(offer[3])} - {safe_display(offer[4])}": offer[0]
                    for offer in offers
                }

                selected_offer_label = st.selectbox("Sélectionnez une offre", list(offer_options.keys()))
                selected_offer_id = offer_options[selected_offer_label]
                selected_offer = load_offer_by_id(selected_offer_id)

                if selected_offer:
                    offer_id, title, company_name, location, contract_type, description, requirements, created_at = selected_offer

                    title = safe_display(title)
                    company_name = safe_display(company_name)
                    location = safe_display(location)
                    contract_type = safe_display(contract_type)
                    description = safe_display(description)
                    requirements = safe_display(requirements)

                    st.markdown(f"""
                    <div class="job-card">
                        <h3>{title}</h3>
                        <p><b>Entreprise :</b> {company_name}</p>
                        <p><b>Lieu :</b> {location}</p>
                        <p><b>Contrat :</b> <span class="badge">{contract_type}</span></p>
                    </div>
                    """, unsafe_allow_html=True)

                    candidates = load_candidates_by_offer(selected_offer_id)

                    if not candidates:
                        st.warning("Aucun candidat pour cette offre.")
                    else:
                        df_candidates = pd.DataFrame(
                            candidates,
                            columns=[
                                "id", "candidate_name", "candidate_email", "phone", "linkedin",
                                "availability", "salary_expectation", "experience_level",
                                "desired_contract", "job_title", "motivation", "file_name",
                                "file_path", "content", "file_data", "status", "created_at"
                            ]
                        )

                        st.dataframe(
                            df_candidates[
                                [
                                    "candidate_name", "candidate_email", "phone",
                                    "availability", "salary_expectation", "experience_level",
                                    "desired_contract", "file_name", "status", "created_at"
                                ]
                            ],
                            use_container_width=True
                        )

                        st.subheader("Télécharger un CV")

                        cv_options = [
                            f"{row['candidate_name']} - {row['file_name']}"
                            for _, row in df_candidates.iterrows()
                        ]

                        selected_cv_label = st.selectbox(
                            "Sélectionnez un CV",
                            cv_options,
                            key=f"download_cv_{selected_offer_id}"
                        )

                        selected_cv_index = cv_options.index(selected_cv_label)
                        selected_cv_row = df_candidates.iloc[selected_cv_index]

                        if selected_cv_row.get("file_data") is not None:
                            st.download_button(
                                label="📥 Télécharger le CV",
                                data=bytes(selected_cv_row["file_data"]),
                                file_name=selected_cv_row["file_name"],
                                mime="application/pdf",
                                key=f"download_button_{selected_cv_row['id']}"
                            )
                        else:
                            st.warning("CV non disponible en base.")

                        if st.button("Analyser les candidats de cette offre"):
                            results = []

                            with st.spinner("Analyse IA en cours..."):
                                full_offer_text = f"""
Titre : {title}
Entreprise : {company_name}
Lieu : {location}
Contrat : {contract_type}

Description :
{description}

Compétences requises :
{requirements}
"""

                                for _, candidate in df_candidates.iterrows():
                                    cv_id = int(candidate["id"])
                                    candidate_name = candidate["candidate_name"]
                                    candidate_email = candidate["candidate_email"]
                                    file_name = candidate["file_name"]
                                    cv_text = candidate["content"]

                                    result = analyze_candidate(full_offer_text, cv_text, file_name)

                                    result["candidate_name"] = candidate_name
                                    result["candidate_email"] = candidate_email
                                    result["file_name"] = file_name
                                    result["cv_id"] = cv_id

                                    save_score(
                                        job_offer_id=selected_offer_id,
                                        cv_id=cv_id,
                                        result=result
                                    )

                                    if result["final_score"] < 45:
                                        update_candidate_status(cv_id, "Refusé")
                                    elif result["final_score"] < 70:
                                        update_candidate_status(cv_id, "À étudier")
                                    else:
                                        update_candidate_status(cv_id, "À contacter")

                                    results.append(result)

                            if results:
                                df = pd.DataFrame(results)
                                df = df.sort_values(by="final_score", ascending=False).reset_index(drop=True)
                                st.session_state.df_results = df
                                st.session_state.selected_result_offer_id = selected_offer_id
                                st.success("Analyse terminée 🎉")

                        if (
                            st.session_state.df_results is not None
                            and st.session_state.selected_result_offer_id == selected_offer_id
                        ):
                            df = st.session_state.df_results

                            st.subheader("Résultats de scoring")

                            col1, col2, col3 = st.columns(3)
                            col1.metric("Nombre de candidats", len(df))
                            col2.metric("Score moyen", int(df["final_score"].mean()))
                            col3.metric("Meilleur score", int(df["final_score"].max()))

                            st.dataframe(df, use_container_width=True)

                            candidate_options = [
                                f"{row['candidate_name']} - {row['final_score']}/100 - {row['file_name']}"
                                for _, row in df.iterrows()
                            ]

                            selected_candidate = st.selectbox("Sélectionnez un candidat", candidate_options)
                            selected_index = candidate_options.index(selected_candidate)
                            row = df.iloc[selected_index]

                            st.subheader("Détail du candidat")

                            colA, colB, colC = st.columns(3)
                            colA.metric("Score IA", f"{row['score']}/100")
                            colB.metric("Similarité", f"{row['similarity']}%")
                            colC.metric("Score final", f"{row['final_score']}/100")

                            st.write(f"**Nom :** {row['candidate_name']}")
                            st.write(f"**Email :** {row.get('candidate_email', '')}")
                            st.write(f"**Fichier :** {row['file_name']}")

                            final_score = int(row.get("final_score", 0))

                            if final_score < 45:
                                st.error("Statut : Refusé ❌")
                            elif final_score < 70:
                                st.warning("Statut : À étudier ⚠️")
                            else:
                                st.success("Statut : À contacter ✅")

                            st.markdown("**Résumé du profil :**")
                            st.write(row.get("summary", ""))

                            st.markdown("**Points forts :**")
                            st.write(row.get("strengths", ""))

                            st.markdown("**Points faibles :**")
                            st.write(row.get("weaknesses", ""))

                            st.markdown("**Recommandation finale :**")
                            st.write(row.get("recommendation", ""))

                            st.markdown("**Questions d’entretien proposées :**")
                            st.write(row.get("interview_questions", ""))

                            st.markdown("**Explication :**")
                            st.write(row.get("explanation", ""))

                            st.subheader("Réponse automatique au candidat")

                            candidate_name = row.get("candidate_name", "Madame, Monsieur")
                            candidate_email = row.get("candidate_email", "")

                            if final_score < 45:
                                subject = "Réponse à votre candidature"
                                message = f"""Bonjour {candidate_name},

Nous vous remercions pour l’intérêt que vous avez porté à notre offre et pour le temps consacré à votre candidature.

Après analyse de votre profil, nous sommes au regret de vous informer que votre candidature n’a pas été retenue pour la suite du processus.

Votre parcours présente des éléments intéressants, mais il ne correspond pas suffisamment aux critères attendus pour ce poste.

Nous vous souhaitons une bonne continuation dans vos recherches professionnelles.

Cordialement,
{FOUNDER_NAME}
{COMPANY_NAME}
"""
                                st.warning("Réponse automatique de refus proposée.")
                            else:
                                subject = "Suite à votre candidature"
                                message = f"""Bonjour {candidate_name},

Nous vous remercions pour votre candidature.

Après analyse de votre profil, votre CV présente une bonne adéquation avec les critères du poste.

Nous souhaiterions échanger avec vous afin d’en savoir plus sur votre parcours et vos disponibilités.

Seriez-vous disponible prochainement pour un échange ?

Cordialement,
{FOUNDER_NAME}
{COMPANY_NAME}
"""
                                st.success("Candidat à contacter.")

                            st.text_area("Message", value=message, height=260)

                            if candidate_email:
                                mailto_link = (
                                    f"mailto:{candidate_email}"
                                    f"?subject={quote(subject)}"
                                    f"&body={quote(message)}"
                                )
                                st.markdown(f"[Envoyer le mail]({mailto_link})")

                            csv = df.to_csv(index=False).encode("utf-8")

                            st.download_button(
                                label="Télécharger les résultats en CSV",
                                data=csv,
                                file_name="resultats_scoring_cv.csv",
                                mime="text/csv"
                            )

        with recruteur_tab4:
            st.markdown('<div class="main-title">Candidatures spontanées</div>', unsafe_allow_html=True)

            spontaneous = load_spontaneous_candidates()

            if not spontaneous:
                st.info("Aucune candidature spontanée reçue.")
            else:
                df_spontaneous = pd.DataFrame(
                    spontaneous,
                    columns=[
                        "id", "candidate_name", "candidate_email", "phone", "linkedin",
                        "availability", "salary_expectation", "experience_level",
                        "desired_contract", "job_title", "motivation", "file_name",
                        "file_path", "content", "file_data", "status", "created_at"
                    ]
                )

                st.dataframe(
                    df_spontaneous[
                        [
                            "candidate_name", "candidate_email", "phone",
                            "job_title", "availability", "salary_expectation",
                            "experience_level", "desired_contract", "file_name", "status", "created_at"
                        ]
                    ],
                    use_container_width=True
                )

                selected_spontaneous = st.selectbox(
                    "Sélectionnez une candidature spontanée",
                    [
                        f"{row['candidate_name']} - {row['job_title']} - {row['file_name']}"
                        for _, row in df_spontaneous.iterrows()
                    ]
                )

                index_spontaneous = [
                    f"{row['candidate_name']} - {row['job_title']} - {row['file_name']}"
                    for _, row in df_spontaneous.iterrows()
                ].index(selected_spontaneous)

                row_sp = df_spontaneous.iloc[index_spontaneous]

                st.subheader("Détail de la candidature spontanée")
                st.write(f"**Nom :** {row_sp['candidate_name']}")
                st.write(f"**Email :** {row_sp['candidate_email']}")
                st.write(f"**Téléphone :** {row_sp['phone']}")
                st.write(f"**LinkedIn :** {row_sp['linkedin']}")
                st.write(f"**Poste recherché :** {row_sp['job_title']}")
                st.write(f"**Disponibilité :** {row_sp['availability']}")
                st.write(f"**Prétentions salariales :** {row_sp['salary_expectation']}")
                st.write(f"**Niveau d’expérience :** {row_sp['experience_level']}")
                st.write(f"**Contrat recherché :** {row_sp['desired_contract']}")
                st.write(f"**Motivation :** {row_sp['motivation']}")
                st.write(f"**Fichier CV :** {row_sp['file_name']}")

                if row_sp.get("file_data") is not None:
                    st.download_button(
                        label="📥 Télécharger le CV",
                        data=bytes(row_sp["file_data"]),
                        file_name=row_sp["file_name"],
                        mime="application/pdf",
                        key=f"download_spontaneous_{row_sp['id']}"
                    )
                else:
                    st.warning("CV non disponible en base.")

        with recruteur_tab5:
            st.markdown('<div class="main-title">Toutes les candidatures</div>', unsafe_allow_html=True)

            all_candidates = load_all_candidates()

            if not all_candidates:
                st.info("Aucune candidature trouvée.")
            else:
                df_all = pd.DataFrame(
                    all_candidates,
                    columns=[
                        "id", "candidate_name", "candidate_email", "phone", "linkedin",
                        "availability", "salary_expectation", "experience_level",
                        "desired_contract", "job_title", "motivation", "file_name",
                        "file_path", "content", "file_data", "status", "job_offer_id", "created_at"
                    ]
                )

                st.subheader("Vue globale des candidatures")

                col_filter1, col_filter2, col_filter3 = st.columns(3)

                with col_filter1:
                    search_candidate = st.text_input("🔎 Rechercher", placeholder="Nom, email, poste...")

                with col_filter2:
                    status_filter = st.selectbox(
                        "Statut",
                        ["Tous"] + sorted([x for x in df_all["status"].dropna().unique().tolist()])
                    )

                with col_filter3:
                    contract_filter = st.selectbox(
                        "Contrat recherché",
                        ["Tous"] + sorted([x for x in df_all["desired_contract"].dropna().unique().tolist()])
                    )

                df_filtered = df_all.copy()

                if search_candidate:
                    search_lower = search_candidate.lower()
                    df_filtered = df_filtered[
                        df_filtered.apply(
                            lambda row: search_lower in " ".join(row.astype(str)).lower(),
                            axis=1
                        )
                    ]

                if status_filter != "Tous":
                    df_filtered = df_filtered[df_filtered["status"] == status_filter]

                if contract_filter != "Tous":
                    df_filtered = df_filtered[df_filtered["desired_contract"] == contract_filter]

                total_count = len(df_all)
                filtered_count = len(df_filtered)
                spontaneous_count = int(df_all["job_offer_id"].isna().sum())
                offer_count = total_count - spontaneous_count

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total candidatures", total_count)
                k2.metric("Résultats filtrés", filtered_count)
                k3.metric("Candidatures offres", offer_count)
                k4.metric("Spontanées", spontaneous_count)

                st.dataframe(
                    df_filtered[
                        [
                            "candidate_name", "candidate_email", "phone", "job_title",
                            "availability", "salary_expectation", "experience_level",
                            "desired_contract", "status", "job_offer_id", "created_at"
                        ]
                    ],
                    use_container_width=True
                )

                csv_all = df_filtered.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Télécharger les candidatures filtrées en CSV",
                    data=csv_all,
                    file_name="toutes_les_candidatures.csv",
                    mime="text/csv"
                )