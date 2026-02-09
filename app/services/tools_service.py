from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from functools import lru_cache
import base64
import html
import ipaddress
from io import BytesIO
import json
import logging
import mimetypes
import os
import re
import socket
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse, urlunparse
import defusedxml.ElementTree as ET
from zipfile import ZipFile

from app.schemas.tools import (
    ExtractJobRequest,
    ExtractJobResponse,
    ExtractResumeUrlRequest,
    ExtractResumeUrlResponse,
    ExtractTextResponse,
    FixPlanItem,
    LeadCaptureRequest,
    Recommendation,
    ResumeFileMeta,
    ResumeLayoutProfile,
    RiskItem,
    ScoreCard,
    SummarizerRequest,
    SummarizerResponse,
    ToolRequest,
    ToolResponse,
    VpnProbeEnrichRequest,
    VpnProbeEnrichResponse,
    VpnProbeGeoRecord,
    VpnToolCard,
    VpnToolVerdict,
    VpnToolRequest,
    VpnToolResponse,
)
from app.services.tools_llm import (
    ToolsLLMError,
    json_completion,
    json_completion_required,
    tools_llm_enabled,
    transcribe_media,
    vision_extract_text,
)

logger = logging.getLogger(__name__)


class QualityEnforcementError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


ProgressCallback = Callable[[dict[str, Any]], None]

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#./-]{1,}")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")
YEARS_RE = re.compile(r"(\d{1,2})\+?\s*(?:years|yrs|year)")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
IPV6_RE = re.compile(r"\b(?:[A-Fa-f0-9]{0,4}:){2,7}[A-Fa-f0-9]{0,4}\b")
HTML_TAG_RE = re.compile(r"<[^>]+>")

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "your", "you", "from", "into", "our", "are",
    "will", "must", "have", "has", "job", "role", "team", "work", "using", "use", "experience",
    "ability", "strong", "required", "preferred", "skills", "skill",
}

TOOL_TERMS = {
    "python", "java", "c#", "c++", "react", "next.js", "nextjs", "node.js", "node", "docker",
    "kubernetes", "aws", "azure", "gcp", "sql", "postgresql", "mongodb", "redis", "tensorflow", "pytorch",
}

DOMAIN_TERMS = {
    "fintech", "healthcare", "ecommerce", "saas", "crm", "erp", "compliance",
    "regulatory", "telecom", "education", "logistics",
}

HARD_FILTER_TERMS = {
    "citizenship", "citizen", "security clearance", "clearance", "visa",
    "work authorization", "authorized", "bachelor", "master", "phd", "degree",
}

LOW_SIGNAL_TERMS = {
    "role", "roles", "team", "product", "technical", "business", "company", "client", "customers", "customer",
    "day", "days", "month", "months", "year", "years", "week", "weeks",
    "required", "requirement", "requirements", "requires", "preferred", "must", "need", "needed",
    "looking", "seeking", "candidate", "candidates", "position", "job", "responsibilities", "responsibility",
    "work", "working", "ability", "abilities", "skill", "skills", "experience",
    "engineer", "engineering", "developer", "development", "platform",
    "plus", "nice", "bonus", "good", "great", "strong", "excellent", "knowledge", "understanding",
    "onsite", "on-site", "hybrid", "remote", "office", "location", "based",
    "full", "time", "part", "level", "senior", "junior", "mid",
    "across", "within", "using", "with", "without", "from", "into",
}

WORK_MODE_TERMS = {"remote", "hybrid", "onsite", "on-site"}

ROLE_SIGNAL_TERMS = {
    "backend", "frontend", "full-stack", "fullstack", "devops", "sre", "qa",
    "architecture", "microservices", "api", "apis", "distributed", "scalable", "performance",
}

SOFT_SKILL_TERMS = {
    "communication",
    "collaboration",
    "leadership",
    "ownership",
    "stakeholder",
    "mentoring",
    "problem-solving",
    "problem",
    "adaptability",
    "teamwork",
    "cross-functional",
    "planning",
    "prioritization",
}

AI_CLICHE_TERMS = {
    "i am excited",
    "passionate about",
    "dynamic professional",
    "results-driven",
    "proven track record",
    "leveraged",
    "synergy",
    "fast-paced environment",
    "detail-oriented",
    "team player",
    "hard-working",
}

ATS_HEAVY_ROLE_TERMS = {
    "engineer",
    "developer",
    "software",
    "backend",
    "frontend",
    "full-stack",
    "platform",
    "devops",
    "sre",
    "qa",
    "architect",
    "analyst",
}

CREATIVE_ROLE_TERMS = {
    "designer",
    "graphic",
    "ux",
    "ui",
    "visual",
    "creative",
    "brand",
    "illustrator",
    "art director",
    "motion",
}

JOB_SITE_HINTS = {
    "greenhouse.io": "greenhouse",
    "lever.co": "lever",
    "myworkdayjobs.com": "workday",
    "workday.com": "workday",
    "indeed.com": "indeed",
}

JOB_AUTH_WALL_MARKERS = {
    "sign in to continue",
    "log in to continue",
    "captcha",
    "verify you are human",
    "access denied",
    "enable javascript",
    "cloudflare",
    "authentication required",
}

RESUME_AUTH_WALL_MARKERS = {
    "sign in to continue",
    "log in to continue",
    "captcha",
    "verify you are human",
    "access denied",
    "authentication required",
    "cloudflare",
    "this file has been moved to trash",
}

RESUME_CONTENT_TYPE_EXTENSION_HINTS = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
    "text/markdown": "md",
    "application/rtf": "rtf",
    "text/rtf": "rtf",
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
    "image/bmp": "bmp",
}

LOW_SIGNAL_KEYWORD_TERMS = {
    "days",
    "day",
    "month",
    "months",
    "week",
    "weeks",
    "technical",
    "product",
    "onsite",
    "on-site",
    "hybrid",
    "remote",
    "looking",
    "required",
    "bonus",
}

VAGUE_QUALITY_PHRASES = {
    "tailor your resume",
    "improve your resume",
    "boost your chances",
    "stand out",
    "keyword stuffing",
}

MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        "risk_hard_filter": "Potential hard filter mismatch: {detail}.",
        "risk_keyword_gap": "High-priority job terms are missing from your resume.",
        "risk_parsing": "ATS parsing risk detected. Keep formatting simpler and one-column.",
        "risk_seniority": "Seniority signal may not align with the role level.",
        "risk_evidence_gap": "Claims could be stronger with measurable outcomes.",
        "fix_keywords_title": "Add missing priority keywords with proof",
        "fix_keywords_reason": "Improves recruiter searchability and JD alignment.",
        "fix_parsing_title": "Simplify resume formatting",
        "fix_parsing_reason": "Reduces extraction errors in applicant tracking systems.",
        "fix_evidence_title": "Add outcome-based bullets",
        "fix_evidence_reason": "Use numbers and outcomes to support your claims.",
        "fix_seniority_title": "Tune level signaling in summary and bullets",
        "fix_seniority_reason": "Match tone and responsibility level to the target role.",
        "lead_saved": "Thanks. We saved your request and will follow up by email.",
        "mode_recruiter": "Recruiter skim",
        "mode_hr": "HR mode",
        "mode_technical": "Technical hiring manager",
        "recommend_apply": "Apply as-is",
        "recommend_fix": "Apply after quick fixes",
        "recommend_skip": "Skip due to likely hard filters",
        "insert_skill": "Add '{term}' as a verified skill.",
        "insert_exp": "Add '{term}' in an experience bullet with measurable evidence.",
        "flag_table": "Table-like content may break ATS extraction.",
        "flag_multicol": "Large spacing suggests a multi-column parsing risk.",
        "flag_header": "Header/footer content can be skipped by some ATS readers.",
        "cover_greeting": "Hello,",
        "cover_p1": "I am applying for this role with direct experience in {top_match}. My recent work shows measurable delivery in similar responsibilities.",
        "cover_p2": "This role emphasizes {role_hint}. I tailored my resume to make outcomes clear and strengthened wording around {improvement} without keyword stuffing.",
        "cover_closing": "Regards,",
        "interview_missing_q": "Can you describe a project where you used {term} in production?",
        "interview_missing_r": "This appears in the job description but has limited evidence in your resume.",
        "interview_seniority_q": "Tell us about a high-stakes decision you led end-to-end.",
        "interview_seniority_r": "Interviewers may validate seniority signals for role scope.",
        "interview_fallback_q": "What was your most impactful project in the last 12 months?",
        "interview_fallback_r": "Standard evidence-depth validation.",
        "red_flag_1": "A claim is listed without measurable impact.",
        "red_flag_2": "A required tool appears in the JD but not clearly in experience bullets.",
        "hf_visa": "visa/work authorization",
        "hf_degree": "degree requirement",
        "hf_clearance": "security clearance",
        "framework_star": "STAR",
        "framework_star_tradeoff": "STAR + tradeoff",
    },
    "de": {
        "risk_hard_filter": "Moeglicher harter Filterkonflikt: {detail}.",
        "risk_keyword_gap": "Wichtige Begriffe aus der Stellenanzeige fehlen im Lebenslauf.",
        "risk_parsing": "ATS-Parsing-Risiko erkannt. Einfache einspaltige Struktur empfohlen.",
        "risk_seniority": "Senioritaetssignal passt moeglicherweise nicht zur Rollenstufe.",
        "risk_evidence_gap": "Aussagen sollten staerker mit messbaren Ergebnissen belegt werden.",
        "fix_keywords_title": "Fehlende Schluesselbegriffe mit Nachweis ergaenzen",
        "fix_keywords_reason": "Verbessert Auffindbarkeit und Abgleich mit der Stellenanzeige.",
        "fix_parsing_title": "Lebenslauf-Format vereinfachen",
        "fix_parsing_reason": "Verringert Extraktionsfehler in ATS-Systemen.",
        "fix_evidence_title": "Ergebnisorientierte Bullet Points hinzufuegen",
        "fix_evidence_reason": "Zahlen und Wirkung zur Unterstuetzung Ihrer Aussagen nutzen.",
        "fix_seniority_title": "Level-Signal in Profil und Bullets anpassen",
        "fix_seniority_reason": "Verantwortungsniveau auf die Zielrolle abstimmen.",
        "lead_saved": "Danke. Wir haben Ihre Anfrage gespeichert und melden uns per E-Mail.",
        "mode_recruiter": "Recruiter-Kurzscan",
        "mode_hr": "HR-Modus",
        "mode_technical": "Technischer Hiring Manager",
        "recommend_apply": "Direkt bewerben",
        "recommend_fix": "Nach kurzen Anpassungen bewerben",
        "recommend_skip": "Wegen harter Filter eher ueberspringen",
        "insert_skill": "Fuege '{term}' als verifizierte Kompetenz hinzu.",
        "insert_exp": "Nutze '{term}' in einem Erfahrungs-Bullet mit messbarem Ergebnis.",
        "flag_table": "Tabellenartige Inhalte koennen die ATS-Extraktion stoeren.",
        "flag_multicol": "Grosse Abstaende deuten auf ein Mehrspalten-Risiko hin.",
        "flag_header": "Inhalte in Kopf- oder Fusszeilen koennen uebersehen werden.",
        "cover_greeting": "Hallo,",
        "cover_p1": "Ich bewerbe mich auf diese Rolle mit direkter Erfahrung in {top_match}. Meine letzten Projekte zeigen messbare Ergebnisse bei aehnlichen Aufgaben.",
        "cover_p2": "Die Stelle betont {role_hint}. Ich habe meinen Lebenslauf auf klare Ergebnisse ausgerichtet und Formulierungen zu {improvement} praezisiert, ohne Keyword-Stuffing.",
        "cover_closing": "Viele Gruesse,",
        "interview_missing_q": "Koennen Sie ein Projekt beschreiben, in dem Sie {term} produktiv eingesetzt haben?",
        "interview_missing_r": "Dieser Punkt steht in der Stellenanzeige, ist aber im Lebenslauf nur schwach belegt.",
        "interview_seniority_q": "Berichten Sie von einer kritischen Entscheidung, die Sie End-to-End verantwortet haben.",
        "interview_seniority_r": "Interviewer pruefen damit oft das Senioritaetsniveau fuer den Rollen-Umfang.",
        "interview_fallback_q": "Was war Ihr wirkungsvollstes Projekt in den letzten 12 Monaten?",
        "interview_fallback_r": "Standardpruefung der Tiefe und Nachweisbarkeit.",
        "red_flag_1": "Eine Aussage ist ohne messbaren Impact formuliert.",
        "red_flag_2": "Ein Pflicht-Tool steht in der JD, fehlt aber klar in den Experience-Bullets.",
        "hf_visa": "Visum/Arbeitserlaubnis",
        "hf_degree": "Abschluss-Anforderung",
        "hf_clearance": "Sicherheitsfreigabe",
        "framework_star": "STAR",
        "framework_star_tradeoff": "STAR + Abwaegung",
    },
    "es": {
        "risk_hard_filter": "Posible conflicto de filtro estricto: {detail}.",
        "risk_keyword_gap": "Faltan terminos prioritarios del puesto en tu CV.",
        "risk_parsing": "Riesgo de parsing ATS detectado. Usa formato simple de una columna.",
        "risk_seniority": "La senal de seniority puede no coincidir con el nivel del rol.",
        "risk_evidence_gap": "Tus afirmaciones serian mas fuertes con resultados medibles.",
        "fix_keywords_title": "Agregar palabras clave prioritarias con evidencia",
        "fix_keywords_reason": "Mejora la encontrabilidad y el ajuste con la oferta.",
        "fix_parsing_title": "Simplificar formato del CV",
        "fix_parsing_reason": "Reduce errores de extraccion en sistemas ATS.",
        "fix_evidence_title": "Agregar bullets orientados a resultados",
        "fix_evidence_reason": "Usa metricas e impacto para respaldar tus logros.",
        "fix_seniority_title": "Ajustar senal de nivel en resumen y experiencia",
        "fix_seniority_reason": "Alinear tono y responsabilidad con el rol objetivo.",
        "lead_saved": "Gracias. Guardamos tu solicitud y te contactaremos por correo.",
        "mode_recruiter": "Lectura rapida recruiter",
        "mode_hr": "Modo RRHH",
        "mode_technical": "Modo hiring manager tecnico",
        "recommend_apply": "Aplicar tal como esta",
        "recommend_fix": "Aplicar despues de ajustes rapidos",
        "recommend_skip": "Omitir por probables filtros estrictos",
        "insert_skill": "Agrega '{term}' como habilidad verificada.",
        "insert_exp": "Incluye '{term}' en una experiencia con evidencia medible.",
        "flag_table": "El contenido tipo tabla puede romper la extraccion ATS.",
        "flag_multicol": "El espaciado amplio sugiere riesgo de parseo multicolumna.",
        "flag_header": "El contenido en encabezado o pie puede ignorarse en algunos ATS.",
        "cover_greeting": "Hola,",
        "cover_p1": "Me postulo a este rol con experiencia directa en {top_match}. Mi trabajo reciente muestra resultados medibles en responsabilidades similares.",
        "cover_p2": "Este rol enfatiza {role_hint}. Ajuste mi CV para mostrar resultados claros y reforzar el lenguaje sobre {improvement} sin relleno de keywords.",
        "cover_closing": "Saludos,",
        "interview_missing_q": "Puedes describir un proyecto donde usaste {term} en produccion?",
        "interview_missing_r": "Esto aparece en la oferta, pero tiene poca evidencia en tu CV.",
        "interview_seniority_q": "Cuentanos una decision critica que lideraste de inicio a fin.",
        "interview_seniority_r": "Los entrevistadores suelen validar asi la seniority para el alcance del rol.",
        "interview_fallback_q": "Cual fue tu proyecto de mayor impacto en los ultimos 12 meses?",
        "interview_fallback_r": "Validacion estandar de profundidad y evidencia.",
        "red_flag_1": "Hay una afirmacion sin impacto medible.",
        "red_flag_2": "Una herramienta requerida esta en la JD, pero no se ve clara en experiencia.",
        "hf_visa": "visa/autorizacion de trabajo",
        "hf_degree": "requisito de titulo",
        "hf_clearance": "habilitacion de seguridad",
        "framework_star": "STAR",
        "framework_star_tradeoff": "STAR + compensaci\u00f3n",
    },
    "fr": {
        "risk_hard_filter": "Conflit possible de filtre strict : {detail}.",
        "risk_keyword_gap": "Des termes prioritaires de l'offre manquent dans le CV.",
        "risk_parsing": "Risque de parsing ATS detecte. Gardez un format simple a une colonne.",
        "risk_seniority": "Le signal de seniorite peut ne pas correspondre au poste cible.",
        "risk_evidence_gap": "Le CV gagnerait en credibilite avec des resultats mesurables.",
        "fix_keywords_title": "Ajouter les mots-cles prioritaires avec preuve",
        "fix_keywords_reason": "Ameliore la trouvabilite et l'alignement avec l'offre.",
        "fix_parsing_title": "Simplifier la mise en forme du CV",
        "fix_parsing_reason": "Reduit les erreurs d'extraction dans les ATS.",
        "fix_evidence_title": "Ajouter des bullets axes resultats",
        "fix_evidence_reason": "Appuyez vos affirmations avec chiffres et impact concret.",
        "fix_seniority_title": "Ajuster le signal de seniorite",
        "fix_seniority_reason": "Aligner niveau de responsabilite et role vise.",
        "lead_saved": "Merci. Votre demande est enregistree et un suivi par e-mail est prevu.",
        "mode_recruiter": "Lecture rapide recruteur",
        "mode_hr": "Mode RH",
        "mode_technical": "Mode manager technique",
        "recommend_apply": "Postuler tel quel",
        "recommend_fix": "Postuler apres corrections rapides",
        "recommend_skip": "A eviter a cause de filtres stricts probables",
        "insert_skill": "Ajoutez '{term}' comme competence verifiee.",
        "insert_exp": "Ajoutez '{term}' dans un bullet d'experience avec preuve mesurable.",
        "flag_table": "Le contenu en tableau peut casser l'extraction ATS.",
        "flag_multicol": "Un espacement important suggere un risque de parsing multicolonne.",
        "flag_header": "Le contenu d'en-tete ou pied de page peut etre ignore.",
        "cover_greeting": "Bonjour,",
        "cover_p1": "Je candidate a ce poste avec une experience directe en {top_match}. Mes projets recents montrent des resultats mesurables sur des responsabilites similaires.",
        "cover_p2": "Ce poste met l'accent sur {role_hint}. J'ai adapte mon CV pour clarifier les resultats et renforcer la formulation autour de {improvement} sans keyword stuffing.",
        "cover_closing": "Cordialement,",
        "interview_missing_q": "Pouvez-vous decrire un projet ou vous avez utilise {term} en production ?",
        "interview_missing_r": "Ceci apparait dans l'offre, mais les preuves sont faibles dans votre CV.",
        "interview_seniority_q": "Parlez-nous d'une decision critique que vous avez menee de bout en bout.",
        "interview_seniority_r": "Les recruteurs valident souvent le niveau de seniorite avec ce type de question.",
        "interview_fallback_q": "Quel a ete votre projet le plus impactant sur les 12 derniers mois ?",
        "interview_fallback_r": "Validation standard de profondeur et de preuve.",
        "red_flag_1": "Une affirmation est presente sans impact mesurable.",
        "red_flag_2": "Un outil requis est dans la JD mais pas clairement visible dans l'experience.",
        "hf_visa": "visa/autorisation de travail",
        "hf_degree": "exigence de diplome",
        "hf_clearance": "habilitation de securite",
        "framework_star": "STAR",
        "framework_star_tradeoff": "STAR + arbitrage",
    },
    "it": {
        "risk_hard_filter": "Possibile conflitto con filtro rigido: {detail}.",
        "risk_keyword_gap": "Mancano termini prioritari della job description nel CV.",
        "risk_parsing": "Rischio parsing ATS rilevato. Usa formato semplice a colonna singola.",
        "risk_seniority": "Il segnale di seniority potrebbe non combaciare con il livello del ruolo.",
        "risk_evidence_gap": "Le affermazioni sono piu solide con risultati misurabili.",
        "fix_keywords_title": "Aggiungi keyword prioritarie con prova",
        "fix_keywords_reason": "Migliora trovabilita e allineamento con la job description.",
        "fix_parsing_title": "Semplifica il formato del CV",
        "fix_parsing_reason": "Riduce errori di estrazione nei sistemi ATS.",
        "fix_evidence_title": "Aggiungi bullet orientati ai risultati",
        "fix_evidence_reason": "Usa numeri e impatto per rafforzare il profilo.",
        "fix_seniority_title": "Regola il segnale di livello",
        "fix_seniority_reason": "Allinea responsabilita e tono al ruolo target.",
        "lead_saved": "Grazie. Abbiamo salvato la richiesta e ti contatteremo via email.",
        "mode_recruiter": "Lettura rapida recruiter",
        "mode_hr": "Modalita HR",
        "mode_technical": "Modalita manager tecnico",
        "recommend_apply": "Candidati subito",
        "recommend_fix": "Candidati dopo correzioni rapide",
        "recommend_skip": "Meglio evitare per probabili filtri rigidi",
        "insert_skill": "Aggiungi '{term}' come competenza verificata.",
        "insert_exp": "Inserisci '{term}' in un bullet di esperienza con prova misurabile.",
        "flag_table": "Contenuti in stile tabella possono rompere l'estrazione ATS.",
        "flag_multicol": "Spaziatura ampia suggerisce rischio di parsing multi-colonna.",
        "flag_header": "Contenuti in header/footer possono essere ignorati da alcuni ATS.",
        "cover_greeting": "Ciao,",
        "cover_p1": "Mi candido per questo ruolo con esperienza diretta in {top_match}. Il mio lavoro recente mostra risultati misurabili su responsabilita simili.",
        "cover_p2": "Il ruolo enfatizza {role_hint}. Ho adattato il CV per rendere chiari i risultati e migliorare il linguaggio su {improvement} senza keyword stuffing.",
        "cover_closing": "Cordiali saluti,",
        "interview_missing_q": "Puoi descrivere un progetto in cui hai usato {term} in produzione?",
        "interview_missing_r": "Questo compare nella job description ma ha poca evidenza nel CV.",
        "interview_seniority_q": "Raccontaci una decisione ad alto impatto che hai guidato end-to-end.",
        "interview_seniority_r": "Gli intervistatori spesso verificano cosi il livello di seniority.",
        "interview_fallback_q": "Qual e stato il tuo progetto con maggiore impatto negli ultimi 12 mesi?",
        "interview_fallback_r": "Validazione standard della profondita e delle prove.",
        "red_flag_1": "Una dichiarazione e presente senza impatto misurabile.",
        "red_flag_2": "Uno strumento richiesto e nella JD ma non e chiaro nei bullet di esperienza.",
        "hf_visa": "visto/autorizzazione al lavoro",
        "hf_degree": "requisito di laurea",
        "hf_clearance": "nulla osta di sicurezza",
        "framework_star": "STAR",
        "framework_star_tradeoff": "STAR + compromesso",
    },
    "ar": {
        "risk_hard_filter": "\u0627\u062d\u062a\u0645\u0627\u0644 \u0639\u062f\u0645 \u062a\u0637\u0627\u0628\u0642 \u0627\u0644\u0641\u0644\u062a\u0631 \u0627\u0644\u062b\u0627\u0628\u062a: {detail}.",
        "risk_keyword_gap": "\u0634\u0631\u0648\u0637 \u0627\u0644\u0648\u0638\u064a\u0641\u0629 \u0630\u0627\u062a \u0627\u0644\u0623\u0648\u0644\u0648\u064a\u0629 \u0627\u0644\u0639\u0627\u0644\u064a\u0629 \u0645\u0641\u0642\u0648\u062f\u0629 \u0645\u0646 \u0633\u064a\u0631\u062a\u0643 \u0627\u0644\u0630\u0627\u062a\u064a\u0629.",
        "risk_parsing": "\u062a\u0645 \u0627\u0643\u062a\u0634\u0627\u0641 \u062e\u0637\u0631 \u062a\u062d\u0644\u064a\u0644 ATS. \u062d\u0627\u0641\u0638 \u0639\u0644\u0649 \u0627\u0644\u062a\u0646\u0633\u064a\u0642 \u0628\u0634\u0643\u0644 \u0623\u0628\u0633\u0637 \u0648\u0639\u0645\u0648\u062f \u0648\u0627\u062d\u062f.",
        "risk_seniority": "\u0642\u062f \u0644\u0627 \u062a\u062a\u0648\u0627\u0641\u0642 \u0625\u0634\u0627\u0631\u0629 \u0627\u0644\u0623\u0642\u062f\u0645\u064a\u0629 \u0645\u0639 \u0645\u0633\u062a\u0648\u0649 \u0627\u0644\u062f\u0648\u0631.",
        "risk_evidence_gap": "\u064a\u0645\u0643\u0646 \u0623\u0646 \u062a\u0643\u0648\u0646 \u0627\u0644\u0645\u0637\u0627\u0644\u0628\u0627\u062a \u0623\u0642\u0648\u0649 \u0645\u0639 \u0646\u062a\u0627\u0626\u062c \u0642\u0627\u0628\u0644\u0629 \u0644\u0644\u0642\u064a\u0627\u0633.",
        "fix_keywords_title": "\u0623\u0636\u0641 \u0627\u0644\u0643\u0644\u0645\u0627\u062a \u0627\u0644\u0631\u0626\u064a\u0633\u064a\u0629 \u0630\u0627\u062a \u0627\u0644\u0623\u0648\u0644\u0648\u064a\u0629 \u0627\u0644\u0645\u0641\u0642\u0648\u062f\u0629 \u0645\u0639 \u0627\u0644\u062f\u0644\u064a\u0644",
        "fix_keywords_reason": "\u064a\u062d\u0633\u0646 \u0625\u0645\u0643\u0627\u0646\u064a\u0629 \u0627\u0644\u0628\u062d\u062b \u0639\u0646 \u0627\u0644\u0645\u062c\u0646\u062f \u0648\u0645\u0648\u0627\u0621\u0645\u0629 JD.",
        "fix_parsing_title": "\u062a\u0628\u0633\u064a\u0637 \u062a\u0646\u0633\u064a\u0642 \u0627\u0644\u0633\u064a\u0631\u0629 \u0627\u0644\u0630\u0627\u062a\u064a\u0629",
        "fix_parsing_reason": "\u064a\u0642\u0644\u0644 \u0645\u0646 \u0623\u062e\u0637\u0627\u0621 \u0627\u0644\u0627\u0633\u062a\u062e\u0631\u0627\u062c \u0641\u064a \u0623\u0646\u0638\u0645\u0629 \u062a\u062a\u0628\u0639 \u0627\u0644\u0645\u062a\u0642\u062f\u0645\u064a\u0646.",
        "fix_evidence_title": "\u0625\u0636\u0627\u0641\u0629 \u0627\u0644\u0631\u0645\u0648\u0632 \u0627\u0644\u0646\u0642\u0637\u064a\u0629 \u0627\u0644\u0645\u0633\u062a\u0646\u062f\u0629 \u0625\u0644\u0649 \u0627\u0644\u0646\u062a\u0627\u0626\u062c",
        "fix_evidence_reason": "\u0627\u0633\u062a\u062e\u062f\u0645 \u0627\u0644\u0623\u0631\u0642\u0627\u0645 \u0648\u0627\u0644\u0646\u062a\u0627\u0626\u062c \u0644\u062f\u0639\u0645 \u0645\u0637\u0627\u0644\u0628\u0627\u062a\u0643.",
        "fix_seniority_title": "\u0636\u0628\u0637 \u0645\u0633\u062a\u0648\u0649 \u0627\u0644\u0625\u0634\u0627\u0631\u0629 \u0641\u064a \u0627\u0644\u0645\u0644\u062e\u0635 \u0648\u0627\u0644\u0631\u0635\u0627\u0635",
        "fix_seniority_reason": "\u0645\u0637\u0627\u0628\u0642\u0629 \u0627\u0644\u0646\u0628\u0631\u0629 \u0648\u0645\u0633\u062a\u0648\u0649 \u0627\u0644\u0645\u0633\u0624\u0648\u0644\u064a\u0629 \u0645\u0639 \u0627\u0644\u062f\u0648\u0631 \u0627\u0644\u0645\u0633\u062a\u0647\u062f\u0641.",
        "lead_saved": "\u0634\u0643\u0631\u064b\u0627. \u0644\u0642\u062f \u062d\u0641\u0638\u0646\u0627 \u0637\u0644\u0628\u0643 \u0648\u0633\u0646\u062a\u0627\u0628\u0639\u0647 \u0639\u0628\u0631 \u0627\u0644\u0628\u0631\u064a\u062f \u0627\u0644\u0625\u0644\u0643\u062a\u0631\u0648\u0646\u064a.",
        "mode_recruiter": "\u0645\u0642\u0634\u0648\u062f \u0627\u0644\u0645\u062c\u0646\u062f",
        "mode_hr": "\u0648\u0636\u0639 \u0627\u0644\u0645\u0648\u0627\u0631\u062f \u0627\u0644\u0628\u0634\u0631\u064a\u0629",
        "mode_technical": "\u0645\u062f\u064a\u0631 \u0627\u0644\u062a\u0648\u0638\u064a\u0641 \u0627\u0644\u0641\u0646\u064a",
        "recommend_apply": "\u062a\u0637\u0628\u064a\u0642 \u0643\u0645\u0627 \u0647\u0648",
        "recommend_fix": "\u062a\u0646\u0637\u0628\u0642 \u0628\u0639\u062f \u0625\u0635\u0644\u0627\u062d\u0627\u062a \u0633\u0631\u064a\u0639\u0629",
        "recommend_skip": "\u062a\u062e\u0637\u064a \u0628\u0633\u0628\u0628 \u0627\u0644\u0645\u0631\u0634\u062d\u0627\u062a \u0627\u0644\u0635\u0639\u0628\u0629 \u0627\u0644\u0645\u062d\u062a\u0645\u0644\u0629",
        "insert_skill": "\u0623\u0636\u0641 '{term}' \u0643\u0645\u0647\u0627\u0631\u0629 \u062a\u0645 \u0627\u0644\u062a\u062d\u0642\u0642 \u0645\u0646\u0647\u0627.",
        "insert_exp": "\u0623\u0636\u0641 \"{term}\" \u0641\u064a \u0642\u0627\u0626\u0645\u0629 \u0627\u0644\u062e\u0628\u0631\u0629 \u0645\u0639 \u0623\u062f\u0644\u0629 \u0642\u0627\u0628\u0644\u0629 \u0644\u0644\u0642\u064a\u0627\u0633.",
        "flag_table": "\u0642\u062f \u064a\u0624\u062f\u064a \u0627\u0644\u0645\u062d\u062a\u0648\u0649 \u0627\u0644\u0634\u0628\u064a\u0647 \u0628\u0627\u0644\u062c\u062f\u0648\u0644 \u0625\u0644\u0649 \u062a\u0639\u0637\u064a\u0644 \u0639\u0645\u0644\u064a\u0629 \u0627\u0633\u062a\u062e\u0631\u0627\u062c \u0627\u0644\u0645\u0646\u0634\u0637\u0627\u062a \u0627\u0644\u0623\u0645\u0641\u064a\u062a\u0627\u0645\u064a\u0646\u064a\u0629.",
        "flag_multicol": "\u062a\u0634\u064a\u0631 \u0627\u0644\u0645\u0633\u0627\u0641\u0627\u062a \u0627\u0644\u0643\u0628\u064a\u0631\u0629 \u0625\u0644\u0649 \u0648\u062c\u0648\u062f \u062e\u0637\u0631 \u062a\u062d\u0644\u064a\u0644 \u0645\u062a\u0639\u062f\u062f \u0627\u0644\u0623\u0639\u0645\u062f\u0629.",
        "flag_header": "\u064a\u0645\u0643\u0646 \u0644\u0628\u0639\u0636 \u0642\u0631\u0627\u0621 ATS \u062a\u062e\u0637\u064a \u0645\u062d\u062a\u0648\u0649 \u0627\u0644\u0631\u0623\u0633/\u0627\u0644\u062a\u0630\u064a\u064a\u0644.",
        "cover_greeting": "\u0645\u0631\u062d\u0628\u064b\u0627\u060c",
        "cover_p1": "\u0623\u0646\u0627 \u0623\u062a\u0642\u062f\u0645 \u0644\u0647\u0630\u0627 \u0627\u0644\u062f\u0648\u0631 \u0628\u062e\u0628\u0631\u0629 \u0645\u0628\u0627\u0634\u0631\u0629 \u0641\u064a {top_match}. \u064a\u064f\u0638\u0647\u0631 \u0639\u0645\u0644\u064a \u0627\u0644\u0623\u062e\u064a\u0631 \u0625\u0646\u062c\u0627\u0632\u064b\u0627 \u0642\u0627\u0628\u0644\u0627\u064b \u0644\u0644\u0642\u064a\u0627\u0633 \u0641\u064a \u0645\u0633\u0624\u0648\u0644\u064a\u0627\u062a \u0645\u0645\u0627\u062b\u0644\u0629.",
        "cover_p2": "\u064a\u0624\u0643\u062f \u0647\u0630\u0627 \u0627\u0644\u062f\u0648\u0631 \u0639\u0644\u0649 {role_hint}. \u0644\u0642\u062f \u0635\u0645\u0645\u062a \u0633\u064a\u0631\u062a\u064a \u0627\u0644\u0630\u0627\u062a\u064a\u0629 \u0644\u062c\u0639\u0644 \u0627\u0644\u0646\u062a\u0627\u0626\u062c \u0648\u0627\u0636\u062d\u0629 \u0648\u062a\u0639\u0632\u064a\u0632 \u0627\u0644\u0635\u064a\u0627\u063a\u0629 \u062d\u0648\u0644 {improvement} \u062f\u0648\u0646 \u062d\u0634\u0648 \u0627\u0644\u0643\u0644\u0645\u0627\u062a \u0627\u0644\u0631\u0626\u064a\u0633\u064a\u0629.",
        "cover_closing": "\u064a\u0639\u062a\u0628\u0631\u060c",
        "interview_missing_q": "\u0647\u0644 \u064a\u0645\u0643\u0646\u0643 \u0648\u0635\u0641 \u0645\u0634\u0631\u0648\u0639 \u0627\u0633\u062a\u062e\u062f\u0645\u062a \u0641\u064a\u0647 {term} \u0641\u064a \u0627\u0644\u0625\u0646\u062a\u0627\u062c\u061f",
        "interview_missing_r": "\u064a\u0638\u0647\u0631 \u0647\u0630\u0627 \u0641\u064a \u0627\u0644\u0648\u0635\u0641 \u0627\u0644\u0648\u0638\u064a\u0641\u064a \u0648\u0644\u0643\u0646 \u0644\u062f\u064a\u0647 \u0623\u062f\u0644\u0629 \u0645\u062d\u062f\u0648\u062f\u0629 \u0641\u064a \u0633\u064a\u0631\u062a\u0643 \u0627\u0644\u0630\u0627\u062a\u064a\u0629.",
        "interview_seniority_q": "\u0623\u062e\u0628\u0631\u0646\u0627 \u0639\u0646 \u0627\u0644\u0642\u0631\u0627\u0631 \u0639\u0627\u0644\u064a \u0627\u0644\u0645\u062e\u0627\u0637\u0631 \u0627\u0644\u0630\u064a \u0627\u062a\u062e\u0630\u062a\u0647 \u0628\u0634\u0643\u0644 \u0643\u0627\u0645\u0644.",
        "interview_seniority_r": "\u064a\u0645\u0643\u0646 \u0644\u0644\u0628\u0627\u062d\u062b\u064a\u0646 \u0627\u0644\u062a\u062d\u0642\u0642 \u0645\u0646 \u0635\u062d\u0629 \u0625\u0634\u0627\u0631\u0627\u062a \u0627\u0644\u0623\u0642\u062f\u0645\u064a\u0629 \u0644\u0646\u0637\u0627\u0642 \u0627\u0644\u062f\u0648\u0631.",
        "interview_fallback_q": "\u0645\u0627 \u0647\u0648 \u0645\u0634\u0631\u0648\u0639\u0643 \u0627\u0644\u0623\u0643\u062b\u0631 \u062a\u0623\u062b\u064a\u0631\u064b\u0627 \u062e\u0644\u0627\u0644 \u0627\u0644\u0640 12 \u0634\u0647\u0631\u064b\u0627 \u0627\u0644\u0645\u0627\u0636\u064a\u0629\u061f",
        "interview_fallback_r": "\u0627\u0644\u062a\u062d\u0642\u0642 \u0645\u0646 \u0639\u0645\u0642 \u0627\u0644\u0623\u062f\u0644\u0629 \u0627\u0644\u0642\u064a\u0627\u0633\u064a\u0629.",
        "red_flag_1": "\u064a\u062a\u0645 \u0625\u062f\u0631\u0627\u062c \u0627\u0644\u0645\u0637\u0627\u0644\u0628\u0629 \u062f\u0648\u0646 \u062a\u0623\u062b\u064a\u0631 \u0642\u0627\u0628\u0644 \u0644\u0644\u0642\u064a\u0627\u0633.",
        "red_flag_2": "\u062a\u0638\u0647\u0631 \u0627\u0644\u0623\u062f\u0627\u0629 \u0627\u0644\u0645\u0637\u0644\u0648\u0628\u0629 \u0641\u064a JD \u0648\u0644\u0643\u0646 \u0644\u064a\u0633 \u0628\u0634\u0643\u0644 \u0648\u0627\u0636\u062d \u0641\u064a \u0627\u0644\u0631\u0645\u0648\u0632 \u0627\u0644\u0646\u0642\u0637\u064a\u0629 \u0644\u0644\u062a\u062c\u0631\u0628\u0629.",
        "hf_visa": "\u062a\u0623\u0634\u064a\u0631\u0629 / \u062a\u0635\u0631\u064a\u062d \u0627\u0644\u0639\u0645\u0644",
        "hf_degree": "\u0645\u062a\u0637\u0644\u0628\u0627\u062a \u0627\u0644\u062f\u0631\u062c\u0629",
        "hf_clearance": "\u062a\u0635\u0631\u064a\u062d \u0623\u0645\u0646\u064a",
        "framework_star": "STAR",
        "framework_star_tradeoff": "STAR + \u0645\u0642\u0627\u064a\u0636\u0629",
    },
}


def _msg(locale: str, key: str, **kwargs: Any) -> str:
    base = MESSAGES.get(locale, MESSAGES["en"])
    template = base.get(key) or MESSAGES["en"].get(key, key)
    return template.format(**kwargs)


def _locale_language_name(locale: str) -> str:
    return {
        "en": "English",
        "de": "German",
        "es": "Spanish",
        "fr": "French",
        "it": "Italian",
        "ar": "Arabic",
    }.get(locale, "English")


def _strict_llm_required() -> bool:
    raw = (os.getenv("TOOLS_STRICT_LLM") or "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _ensure_llm_ready(tool_slug: str) -> None:
    if not _strict_llm_required():
        return
    if not tools_llm_enabled():
        raise QualityEnforcementError(
            f"AI quality mode is required for '{tool_slug}' but LLM is not configured. "
            "Set OPENAI_API_KEY and keep TOOLS_LLM_ENABLED=true.",
            status_code=503,
        )


def _safe_str(value: Any, max_len: int = 1500) -> str:
    if not isinstance(value, str):
        return ""
    text = re.sub(r"\s+", " ", value).strip()
    if len(text) > max_len:
        text = text[:max_len].rstrip()
    return text


def _safe_str_list(value: Any, max_items: int, max_len: int = 220) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    for item in value:
        text = _safe_str(item, max_len=max_len)
        if text:
            output.append(text)
        if len(output) >= max_items:
            break
    return output


def _contains_vague_quality_phrase(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in VAGUE_QUALITY_PHRASES)


def _ensure_quality_generation(
    *,
    tool_slug: str,
    generation_mode: str,
    generation_scope: str,
    sample_texts: list[str] | None = None,
) -> None:
    if not _strict_llm_required():
        return
    if generation_mode != "llm":
        raise QualityEnforcementError(
            f"AI quality mode is required for '{tool_slug}', but generation was not AI-verified.",
            status_code=503,
        )
    if generation_scope in {"heuristic", "", "fallback"}:
        raise QualityEnforcementError(
            f"AI quality mode is required for '{tool_slug}', but response scope was '{generation_scope}'.",
            status_code=503,
        )
    if sample_texts:
        weak_count = sum(1 for text in sample_texts if not text or _contains_vague_quality_phrase(text))
        if weak_count > 0:
            raise QualityEnforcementError(
                f"AI output quality check failed for '{tool_slug}'. Please retry with clearer input/job description.",
                status_code=422,
            )


def _emit_progress(
    progress_callback: ProgressCallback | None,
    *,
    stage: str,
    label: str,
    percent: int,
    detail: str = "",
) -> None:
    if not progress_callback:
        return
    progress_callback(
        {
            "stage": stage,
            "label": label,
            "percent": _clamp_int(percent, default=0, min_value=0, max_value=100),
            "detail": _safe_str(detail, max_len=240),
            "emitted_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def _line_snippets_with_term(text: str, term: str, *, max_items: int = 2) -> list[str]:
    if not term:
        return []
    lowered_term = term.lower().strip()
    snippets: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if lowered_term not in cleaned.lower():
            continue
        snippets.append(_safe_str(cleaned, max_len=220))
        if len(snippets) >= max_items:
            break
    return snippets


def _term_evidence_maps(
    *,
    resume_text: str,
    jd_text: str,
    matched_terms: list[str],
    missing_terms: list[str],
    hard_filter_hits: list[str],
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, dict[str, list[str]]]]:
    matched_term_evidence: dict[str, list[str]] = {}
    for term in matched_terms[:18]:
        snippets = _line_snippets_with_term(resume_text, term, max_items=3)
        if snippets:
            matched_term_evidence[term] = snippets

    missing_term_context: dict[str, list[str]] = {}
    for term in missing_terms[:18]:
        snippets = _line_snippets_with_term(jd_text, term, max_items=3)
        if snippets:
            missing_term_context[term] = snippets

    hard_filter_evidence: dict[str, dict[str, list[str]]] = {}
    for item in hard_filter_hits:
        key = _safe_str(item, max_len=80)
        if not key:
            continue
        hard_filter_evidence[key] = {
            "jd_snippets": _line_snippets_with_term(jd_text, key, max_items=2),
            "resume_snippets": _line_snippets_with_term(resume_text, key, max_items=2),
        }

    return matched_term_evidence, missing_term_context, hard_filter_evidence


def _safe_question_items(value: Any, max_items: int = 6) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    output: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        question = _safe_str(item.get("question"), max_len=220)
        reason = _safe_str(item.get("reason"), max_len=240)
        framework = _safe_str(item.get("framework"), max_len=60) or "STAR"
        if question and reason:
            output.append({"question": question, "reason": reason, "framework": framework})
        if len(output) >= max_items:
            break
    return output


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text):
        token = raw.lower().strip(".,;:!?()[]{}\"'")
        if token:
            tokens.append(token)
    return tokens


def _important_terms(text: str, limit: int = 40) -> list[str]:
    tokens = [t for t in _tokenize(text) if t not in STOPWORDS and len(t) > 2]
    return [term for term, _ in Counter(tokens).most_common(limit)]


def _looks_numeric_or_noise(term: str) -> bool:
    return bool(re.fullmatch(r"\d+[a-z]*", term)) or term in {"etc", "misc", "various"}


def _looks_like_location_constraint(term: str, jd_lower: str) -> bool:
    if len(term) < 3:
        return False
    patterns = [
        rf"\b(?:in|at|near|based in|located in|onsite in|on-site in)\s+{re.escape(term)}\b",
        rf"\b{re.escape(term)}\s+(?:office|location|region)\b",
    ]
    return any(re.search(pattern, jd_lower) for pattern in patterns)


def _is_actionable_keyword(term: str, jd_lower: str) -> bool:
    if not term:
        return False
    if len(term) < 3:
        return False
    if term in STOPWORDS or term in LOW_SIGNAL_TERMS:
        return False
    if term in WORK_MODE_TERMS:
        return False
    if _looks_numeric_or_noise(term):
        return False
    if _looks_like_location_constraint(term, jd_lower):
        return False
    return True


def _seniority_to_years(value: str) -> int:
    return {
        "junior": 1,
        "mid": 3,
        "senior": 6,
        "lead": 9,
        "career-switcher": 1,
    }.get(value, 3)


VALID_RECOMMENDATIONS: set[str] = {"apply", "fix", "skip"}
VALID_RISK_TYPES: set[str] = {"hard_filter", "keyword_gap", "parsing", "seniority", "evidence_gap"}
VALID_RISK_SEVERITIES: set[str] = {"low", "medium", "high"}


def _clamp_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, parsed))


def _clamp_float(value: Any, default: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, parsed))


def _safe_recommendation(value: Any, default: Recommendation) -> Recommendation:
    rec = _safe_str(value, max_len=24).lower()
    if rec in VALID_RECOMMENDATIONS:
        return rec  # type: ignore[return-value]
    return default


def _safe_risk_items(value: Any) -> list[RiskItem]:
    if not isinstance(value, list):
        return []
    output: list[RiskItem] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        r_type = _safe_str(item.get("type"), max_len=40).lower()
        r_severity = _safe_str(item.get("severity"), max_len=16).lower()
        r_message = _safe_str(item.get("message"), max_len=240)
        if r_type not in VALID_RISK_TYPES or r_severity not in VALID_RISK_SEVERITIES or not r_message:
            continue
        output.append(RiskItem(type=r_type, severity=r_severity, message=r_message))
        if len(output) >= 8:
            break
    return output


def _normalize_fix_id(value: Any, fallback_index: int) -> str:
    raw = _safe_str(value, max_len=60).lower()
    normalized = re.sub(r"[^a-z0-9-]+", "-", raw).strip("-")
    if normalized:
        return normalized
    return f"llm-fix-{fallback_index}"


def _safe_fix_plan_items(value: Any) -> list[FixPlanItem]:
    if not isinstance(value, list):
        return []
    output: list[FixPlanItem] = []
    for idx, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        title = _safe_str(item.get("title"), max_len=120)
        reason = _safe_str(item.get("reason"), max_len=220)
        if not title or not reason:
            continue
        output.append(
            FixPlanItem(
                id=_normalize_fix_id(item.get("id"), idx),
                title=title,
                impact_score=_clamp_int(item.get("impact_score"), default=60, min_value=1, max_value=100),
                effort_minutes=_clamp_int(item.get("effort_minutes"), default=20, min_value=5, max_value=180),
                reason=reason,
            )
        )
        if len(output) >= 8:
            break
    return output


def _safe_vpn_recommendations(value: Any, max_items: int = 4) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        provider = _safe_str(item.get("provider"), max_len=80)
        reason = _safe_str(item.get("reason"), max_len=240)
        best_for = _safe_str(item.get("best_for"), max_len=140)
        caution = _safe_str(item.get("caution"), max_len=180)
        fit_score = _clamp_int(item.get("fit_score"), default=70, min_value=1, max_value=100)
        if not provider or not reason:
            continue
        key = provider.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "provider": provider,
                "reason": reason,
                "best_for": best_for,
                "caution": caution,
                "fit_score": fit_score,
            }
        )
        if len(output) >= max_items:
            break
    return output


def _extract_ip_list(value: Any, max_items: int = 40) -> list[str]:
    if isinstance(value, str):
        candidates = re.split(r"[\s,;]+", value)
    elif isinstance(value, list):
        candidates = [str(item) for item in value]
    else:
        candidates = []

    output: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        candidate = raw.strip()
        if not candidate:
            continue
        if candidate in seen:
            continue
        try:
            ipaddress.ip_address(candidate)
        except ValueError:
            continue
        seen.add(candidate)
        output.append(candidate)
        if len(output) >= max_items:
            break
    return output


def _is_public_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved)


def _country_match(expected: str, actual: str) -> bool:
    lhs = expected.strip().lower()
    rhs = actual.strip().lower()
    if not lhs or not rhs:
        return False
    if lhs == rhs:
        return True
    aliases = {
        "united states": {"usa", "us", "united states of america"},
        "united kingdom": {"uk", "great britain", "britain"},
        "uae": {"united arab emirates"},
    }
    for canonical, values in aliases.items():
        all_values = set(values) | {canonical}
        if lhs in all_values and rhs in all_values:
            return True
    return False


@lru_cache(maxsize=2048)
def _geo_lookup_ip(ip_value: str) -> dict[str, Any]:
    if not _is_public_ip(ip_value):
        return {
            "ip": ip_value,
            "country": None,
            "country_code": None,
            "region": None,
            "city": None,
            "isp": None,
            "source": "private",
            "is_private": True,
        }

    try:
        import httpx
    except Exception:
        return {
            "ip": ip_value,
            "country": None,
            "country_code": None,
            "region": None,
            "city": None,
            "isp": None,
            "source": "unavailable",
            "is_private": False,
        }

    providers = [
        ("ipwhois", f"https://ipwho.is/{ip_value}"),
        ("ipapi", f"https://ipapi.co/{ip_value}/json/"),
    ]
    for provider, url in providers:
        try:
            with httpx.Client(timeout=5.5, follow_redirects=True) as client:
                response = client.get(url)
            if response.status_code >= 400:
                continue
            payload = response.json()
            if not isinstance(payload, dict):
                continue

            if provider == "ipwhois":
                if payload.get("success") is False:
                    continue
                return {
                    "ip": ip_value,
                    "country": _safe_str(payload.get("country"), max_len=120) or None,
                    "country_code": _safe_str(payload.get("country_code"), max_len=10) or None,
                    "region": _safe_str(payload.get("region"), max_len=120) or None,
                    "city": _safe_str(payload.get("city"), max_len=120) or None,
                    "isp": _safe_str(payload.get("connection", {}).get("isp") if isinstance(payload.get("connection"), dict) else None, max_len=180) or None,
                    "source": provider,
                    "is_private": False,
                }

            country_name = _safe_str(payload.get("country_name"), max_len=120) or _safe_str(payload.get("country"), max_len=120)
            if not country_name:
                continue
            return {
                "ip": ip_value,
                "country": country_name,
                "country_code": _safe_str(payload.get("country_code"), max_len=10) or None,
                "region": _safe_str(payload.get("region"), max_len=120) or None,
                "city": _safe_str(payload.get("city"), max_len=120) or None,
                "isp": _safe_str(payload.get("org"), max_len=180) or None,
                "source": provider,
                "is_private": False,
            }
        except Exception:
            continue

    return {
        "ip": ip_value,
        "country": None,
        "country_code": None,
        "region": None,
        "city": None,
        "isp": None,
        "source": "lookup_failed",
        "is_private": False,
    }


def _dominant_country(records: list[dict[str, Any]]) -> str | None:
    counts: Counter[str] = Counter()
    for item in records:
        country = _safe_str(item.get("country"), max_len=120)
        if country:
            counts[country] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _credibility_score(resume_text: str, jd_text: str) -> dict[str, Any]:
    bullets = [line.strip() for line in resume_text.splitlines() if line.strip().startswith(("-", "*", "\u2022"))]
    if not bullets:
        bullets = [line.strip() for line in resume_text.splitlines() if line.strip()]

    evidence_bullets = sum(1 for bullet in bullets if re.search(r"\d", bullet))
    action_bullets = sum(1 for bullet in bullets if re.match(r"^(built|led|delivered|optimized|designed|implemented|migrated|reduced|increased)\b", bullet.lower()))
    weak_claims = [
        bullet for bullet in bullets
        if len(bullet.split()) >= 6 and not re.search(r"\d", bullet) and "responsible for" in bullet.lower()
    ]

    jd_terms = set(_important_terms(jd_text, limit=50))
    resume_terms = set(_important_terms(resume_text, limit=80))
    evidence_alignment = len(jd_terms & resume_terms)

    base = 48
    base += min(24, evidence_bullets * 4)
    base += min(14, action_bullets * 2)
    base += min(10, evidence_alignment)
    base -= min(16, len(weak_claims) * 4)
    score = _clamp_int(base, default=60, min_value=1, max_value=100)
    return {
        "score": score,
        "evidence_bullets": evidence_bullets,
        "action_bullets": action_bullets,
        "weak_claims": weak_claims[:8],
    }


def _keyword_stuffing_report(resume_text: str, target_terms: list[str]) -> dict[str, Any]:
    resume_tokens = _tokenize(resume_text)
    total_tokens = max(1, len(resume_tokens))
    counts = Counter(resume_tokens)
    flags: list[dict[str, Any]] = []
    for term in target_terms[:30]:
        occurrences = counts.get(term, 0)
        if occurrences <= 0:
            continue
        density = occurrences / total_tokens
        if density >= 0.035 or occurrences >= 7:
            flags.append(
                {
                    "term": term,
                    "occurrences": occurrences,
                    "density": round(density, 4),
                    "risk": "high" if density >= 0.06 or occurrences >= 10 else "medium",
                }
            )
    status = "clean"
    if any(item["risk"] == "high" for item in flags):
        status = "high-risk"
    elif flags:
        status = "attention"
    return {"status": status, "flags": flags[:8]}


def _analyze_bullet_quality(resume_text: str) -> dict[str, Any]:
    bullets = [line.strip(" -*\u2022\t") for line in resume_text.splitlines() if line.strip().startswith(("-", "*", "\u2022"))]
    if not bullets:
        bullets = [line.strip() for line in resume_text.splitlines() if line.strip()][:12]

    findings: list[dict[str, Any]] = []
    for bullet in bullets[:14]:
        has_action = bool(re.match(r"^(built|led|delivered|optimized|designed|implemented|migrated|reduced|increased|automated)\b", bullet.lower()))
        has_context = len(_tokenize(bullet)) >= 8
        has_outcome = bool(re.search(r"\d|%|faster|reduced|increased|improved|saved", bullet.lower()))
        score = 35 + (25 if has_action else 0) + (20 if has_context else 0) + (20 if has_outcome else 0)
        findings.append(
            {
                "bullet": bullet,
                "what": has_action,
                "how": has_context,
                "why": has_outcome,
                "quality_score": _clamp_int(score, default=55, min_value=1, max_value=100),
            }
        )
    average = int(round(sum(item["quality_score"] for item in findings) / max(1, len(findings))))
    return {"average_score": average, "items": findings[:8]}


def _humanization_report(text: str) -> dict[str, Any]:
    lowered = text.lower()
    detected = sorted({term for term in AI_CLICHE_TERMS if term in lowered})
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    rewritten_samples: list[dict[str, str]] = []
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        lowered_sentence = s.lower()
        if not any(term in lowered_sentence for term in AI_CLICHE_TERMS):
            continue
        cleaned = re.sub(r"\b(results-driven|dynamic professional|passionate about)\b", "", lowered_sentence, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip().capitalize()
        if cleaned and cleaned != s:
            rewritten_samples.append({"original": s, "suggested": cleaned})
        if len(rewritten_samples) >= 5:
            break
    return {
        "cliche_count": len(detected),
        "detected_cliches": detected,
        "rewrites": rewritten_samples,
        "status": "clean" if not detected else "attention",
    }


def _header_link_density(text: str) -> float:
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()][:6]
    if not lines:
        return 0.0
    link_hits = sum(1 for line in lines if ("http://" in line or "https://" in line or "linkedin.com" in line))
    return round(link_hits / max(1, len(lines)), 2)


def _normalize_extracted_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _infer_layout_profile_from_text(resume_text: str, *, source_type: str = "text") -> dict[str, Any]:
    if not resume_text.strip():
        return {
            "detected_layout": "unknown",
            "column_count": 1,
            "confidence": 0.2,
            "table_count": 0,
            "header_link_density": 0.0,
            "complexity_score": 20,
            "source_type": source_type if source_type in {"pdf", "word", "text", "image"} else "unknown",
            "signals": ["no_resume_text"],
        }

    text_signals = _text_layout_signals(resume_text)
    pipe_hits = _clamp_int(text_signals["pipe_token_total"], default=0, min_value=0, max_value=10000)
    wide_space_hits = _clamp_int(text_signals["wide_space_lines"], default=0, min_value=0, max_value=10000)
    tab_hits = _clamp_int(text_signals["tab_line_count"], default=0, min_value=0, max_value=10000)
    table_word_hits = len(re.findall(r"\btable\b", resume_text.lower()))

    detected_layout = "single_column"
    column_count = 1
    confidence = 0.38 if source_type in {"text", "unknown"} else 0.56
    signals: list[str] = []

    if pipe_hits >= 6:
        signals.append("pipe_delimiters_detected")
    if wide_space_hits >= 5:
        signals.append("wide_spacing_blocks_detected")
    if tab_hits >= 3:
        signals.append("tabular_alignment_detected")
    if text_signals["markdown_table_lines"] > 0:
        signals.append("markdown_table_separator_detected")
    if text_signals["consistent_pipe_pattern"]:
        signals.append("consistent_pipe_table_pattern")
    if text_signals["probable_table"]:
        signals.append("probable_table_structure_detected")

    # Text-only inference should stay conservative: avoid treating header separators
    # (e.g., "A | B | C") as true multi-column evidence.
    if source_type in {"text", "unknown"}:
        if text_signals["probable_table"] and (
            pipe_hits >= 14 or (wide_space_hits >= 9 and tab_hits >= 4)
        ):
            detected_layout = "multi_column"
            column_count = 2
            confidence = max(confidence, 0.76)
            signals.append("multi_column_pattern_detected")
        elif text_signals["probable_table"] or wide_space_hits >= 8:
            detected_layout = "hybrid"
            column_count = 2
            confidence = max(confidence, 0.62)
            signals.append("hybrid_layout_pattern_detected")
    else:
        # For extracted PDF/DOCX text, line spacing noise is common; only mark hybrid from text
        # when table-like structure is actually detected. Multi-column is determined by geometry/XML.
        if text_signals["probable_table"]:
            detected_layout = "hybrid"
            column_count = 2
            confidence = max(confidence, 0.64)
            signals.append("hybrid_layout_pattern_detected")

    table_count = 0
    if text_signals["probable_table"]:
        table_count += 1
    if table_word_hits:
        table_count += 1

    header_density = _header_link_density(resume_text)
    complexity = 18 + (table_count * 18) + min(16, wide_space_hits * 2)
    if text_signals["probable_table"]:
        complexity += min(12, pipe_hits * 2)
    complexity += int(round(header_density * 22))
    if detected_layout == "multi_column":
        complexity += 20
    elif detected_layout == "hybrid":
        complexity += 10

    return {
        "detected_layout": detected_layout,
        "column_count": column_count,
        "confidence": round(min(0.95, max(0.2, confidence)), 2),
        "table_count": _clamp_int(table_count, default=0, min_value=0, max_value=200),
        "header_link_density": round(min(1.0, max(0.0, header_density)), 2),
        "complexity_score": _clamp_int(complexity, default=25, min_value=0, max_value=100),
        "source_type": source_type if source_type in {"pdf", "word", "text", "image"} else "unknown",
        "signals": signals[:20],
    }


def _coerce_layout_profile(raw_profile: Any, resume_text: str) -> dict[str, Any]:
    fallback = _infer_layout_profile_from_text(resume_text, source_type="text")
    if raw_profile is None:
        return fallback

    if isinstance(raw_profile, ResumeLayoutProfile):
        raw = raw_profile.model_dump()
    elif hasattr(raw_profile, "model_dump"):
        raw = raw_profile.model_dump()
    elif isinstance(raw_profile, dict):
        raw = raw_profile
    else:
        return fallback

    detected_layout = _safe_str(raw.get("detected_layout"), max_len=32).lower()
    if detected_layout not in {"single_column", "multi_column", "hybrid", "unknown"}:
        detected_layout = fallback["detected_layout"]

    source_type = _safe_str(raw.get("source_type"), max_len=16).lower()
    if source_type not in {"pdf", "word", "text", "image", "unknown"}:
        source_type = fallback["source_type"]

    signals = raw.get("signals")
    if isinstance(signals, list):
        parsed_signals = [
            _safe_str(signal, max_len=80)
            for signal in signals[:20]
            if _safe_str(signal, max_len=80)
        ]
    else:
        parsed_signals = fallback["signals"]

    coerced = {
        "detected_layout": detected_layout,
        "column_count": _clamp_int(raw.get("column_count"), default=fallback["column_count"], min_value=1, max_value=4),
        "confidence": round(_clamp_float(raw.get("confidence"), default=fallback["confidence"], min_value=0.0, max_value=1.0), 2),
        "table_count": _clamp_int(raw.get("table_count"), default=fallback["table_count"], min_value=0, max_value=200),
        "header_link_density": round(
            _clamp_float(raw.get("header_link_density"), default=fallback["header_link_density"], min_value=0.0, max_value=1.0),
            2,
        ),
        "complexity_score": _clamp_int(raw.get("complexity_score"), default=fallback["complexity_score"], min_value=0, max_value=100),
        "source_type": source_type,
        "signals": parsed_signals,
    }
    effective_layout = _effective_detected_layout(coerced, resume_text)
    if effective_layout != coerced["detected_layout"]:
        coerced["signals"] = list(dict.fromkeys([*coerced.get("signals", []), "layout_downgraded_low_evidence"]))[:20]
        coerced["detected_layout"] = effective_layout
        if effective_layout in {"single_column", "unknown"}:
            coerced["column_count"] = 1
            coerced["confidence"] = round(min(float(coerced.get("confidence", 0.5)), 0.59), 2)
    return coerced


def _coerce_resume_file_meta(raw_meta: Any) -> dict[str, str]:
    if raw_meta is None:
        return {"filename": "", "extension": "", "source_type": "unknown"}
    if isinstance(raw_meta, ResumeFileMeta):
        raw = raw_meta.model_dump()
    elif hasattr(raw_meta, "model_dump"):
        raw = raw_meta.model_dump()
    elif isinstance(raw_meta, dict):
        raw = raw_meta
    else:
        return {"filename": "", "extension": "", "source_type": "unknown"}

    return {
        "filename": _safe_str(raw.get("filename"), max_len=255),
        "extension": _safe_str(raw.get("extension"), max_len=20).lower(),
        "source_type": _safe_str(raw.get("source_type"), max_len=20).lower() or "unknown",
    }


def _layout_fit_for_target(
    *,
    layout_profile: dict[str, Any],
    target_region: str,
    jd_text: str,
    resume_text: str = "",
) -> dict[str, Any]:
    jd_lower = jd_text.lower()
    region = target_region if target_region in {"US", "EU", "UK", "Other"} else "Other"
    ats_heavy_role = any(term in jd_lower for term in ATS_HEAVY_ROLE_TERMS)
    creative_role = any(term in jd_lower for term in CREATIVE_ROLE_TERMS)
    role_profile = "creative" if creative_role else "ats_heavy" if ats_heavy_role else "general"

    strict_penalty = {"US": 18, "UK": 17, "EU": 12, "Other": 10}
    moderate_penalty = {"US": 10, "UK": 9, "EU": 7, "Other": 6}

    detected_layout = _effective_detected_layout(layout_profile, resume_text)
    if detected_layout == "multi_column":
        penalty = strict_penalty[region]
    elif detected_layout == "hybrid":
        penalty = moderate_penalty[region]
    elif detected_layout == "unknown":
        penalty = 5
    else:
        penalty = 0

    if creative_role:
        penalty = int(round(penalty * 0.45))
    elif not ats_heavy_role:
        penalty = int(round(penalty * 0.75))

    if penalty <= 4:
        fit_level = "good"
        severity = "low"
    elif penalty <= 11:
        fit_level = "moderate"
        severity = "medium"
    else:
        fit_level = "poor"
        severity = "high"

    if detected_layout == "single_column":
        format_recommendation = (
            "Single-column resume detected. This format is typically ATS-friendly for this target."
        )
    elif detected_layout == "multi_column":
        format_recommendation = (
            "Multi-column resume detected. Prepare a single-column variant for stronger ATS parsing reliability."
        )
    elif detected_layout == "hybrid":
        format_recommendation = (
            "Hybrid layout detected. Simplifying to one clear reading flow can improve parsing consistency."
        )
    else:
        format_recommendation = (
            "Layout could not be determined with confidence. Prefer a simple single-column structure."
        )

    return {
        "region": region,
        "role_profile": role_profile,
        "fit_level": fit_level,
        "severity": severity,
        "penalty": _clamp_int(penalty, default=0, min_value=0, max_value=30),
        "format_recommendation": format_recommendation,
    }


def _parsing_penalty(
    resume_text: str,
    *,
    layout_profile: dict[str, Any] | None = None,
    layout_fit: dict[str, Any] | None = None,
) -> int:
    resume_lower = resume_text.lower()
    text_signals = _text_layout_signals(resume_text)
    penalty = 0
    if text_signals["probable_table"] or "table" in resume_lower or "graphic" in resume_lower:
        penalty += 8
    if text_signals["wide_space_lines"] >= 10 and text_signals["tab_line_count"] >= 3:
        penalty += 3
    if not EMAIL_RE.search(resume_text):
        penalty += 6
    if not PHONE_RE.search(resume_text):
        penalty += 4

    if layout_profile:
        detected_layout = _effective_detected_layout(layout_profile, resume_text)
        strong_layout = _has_strong_layout_evidence(layout_profile)
        confidence = _clamp_float(layout_profile.get("confidence"), default=0.0, min_value=0.0, max_value=1.0)
        if detected_layout == "multi_column":
            penalty += 8 if strong_layout or confidence >= 0.72 else 4
        elif detected_layout == "hybrid":
            penalty += 4 if strong_layout or confidence >= 0.62 else 2
        complexity_score = _clamp_int(layout_profile.get("complexity_score"), default=20, min_value=0, max_value=100)
        table_count = _clamp_int(layout_profile.get("table_count"), default=0, min_value=0, max_value=200)
        header_density = _clamp_float(layout_profile.get("header_link_density"), default=0.0, min_value=0.0, max_value=1.0)
        complexity_penalty = int(round(complexity_score / 18))
        if not strong_layout:
            complexity_penalty = max(0, complexity_penalty - 2)
        penalty += complexity_penalty
        penalty += min(8, table_count * 3) if strong_layout else min(4, table_count * 2)
        if header_density >= 0.5:
            penalty += 4

    if layout_fit:
        penalty += _clamp_int(layout_fit.get("penalty"), default=0, min_value=0, max_value=30)

    return _clamp_int(penalty, default=0, min_value=0, max_value=80)


def _count_repetition_issues(resume_text: str) -> int:
    lines = [line.strip().lower() for line in resume_text.splitlines() if line.strip()]
    duplicates = sum(count - 1 for _, count in Counter(lines).items() if count > 1)

    starter_counter: Counter[str] = Counter()
    for line in lines:
        if not line.startswith(("-", "*", "•")):
            continue
        words = [token for token in _tokenize(line) if token]
        if words:
            starter_counter[words[0]] += 1
    starter_repetition = sum(max(0, count - 2) for _, count in starter_counter.items() if count >= 3)
    return _clamp_int(duplicates + starter_repetition, default=0, min_value=0, max_value=12)


def _count_spelling_grammar_issues(resume_text: str) -> int:
    suspicious_patterns = [
        len(re.findall(r"\s{2,}", resume_text)),
        len(re.findall(r"\b(?:teh|adress|managment|responsiblity|enviroment)\b", resume_text.lower())),
        len(re.findall(r"\bi\b", resume_text)),
        len(re.findall(r"[!?.,]{2,}", resume_text)),
    ]
    issue_count = sum(suspicious_patterns)
    return _clamp_int(issue_count, default=0, min_value=0, max_value=15)


def _issue_label(issue_count: int) -> str:
    if issue_count <= 0:
        return "No issues"
    if issue_count == 1:
        return "1 issue"
    return f"{issue_count} issues"


def _check_status(issue_count: int) -> str:
    return "ok" if issue_count <= 0 else "issue"


QUANT_IMPACT_KEYWORDS = {
    "latency",
    "throughput",
    "uptime",
    "availability",
    "revenue",
    "cost",
    "conversion",
    "retention",
    "performance",
    "defect",
    "incidents",
    "tickets",
    "users",
    "requests",
    "transactions",
    "deployments",
    "pipelines",
    "automation",
    "sla",
}

IMPACT_VERB_HINTS = {
    "improved",
    "reduced",
    "increased",
    "cut",
    "saved",
    "accelerated",
    "optimized",
    "scaled",
    "drove",
    "boosted",
    "decreased",
    "delivered",
}

COMMON_TYPO_MAP = {
    "teh": "the",
    "adress": "address",
    "managment": "management",
    "responsiblity": "responsibility",
    "enviroment": "environment",
    "recieve": "receive",
    "seperate": "separate",
}

STRONG_LAYOUT_SIGNAL_HINTS = {
    "pdf_two_column_x_bands",
    "docx_section_columns_detected",
    "multi_column_pattern_detected",
}


def _text_layout_signals(resume_text: str) -> dict[str, Any]:
    lines = [line.rstrip() for line in resume_text.splitlines() if line.strip()]
    pipe_lines = [line for line in lines if line.count("|") >= 2]
    pipe_counts = [line.count("|") for line in pipe_lines]
    pipe_token_total = sum(pipe_counts)
    wide_space_lines = sum(1 for line in lines if re.search(r"\s{4,}", line))
    tab_line_count = sum(1 for line in lines if "\t" in line)
    markdown_table_lines = sum(
        1 for line in lines
        if re.match(r"^\s*\|?\s*:?-{2,}\s*\|", line) and "|" in line
    )

    consistent_pipe_pattern = False
    if len(pipe_counts) >= 3:
        spread = max(pipe_counts) - min(pipe_counts)
        average = sum(pipe_counts) / max(1, len(pipe_counts))
        consistent_pipe_pattern = spread <= 2 and average >= 3

    probable_table = (
        markdown_table_lines >= 1
        or (len(pipe_lines) >= 3 and consistent_pipe_pattern and pipe_token_total >= 10)
        or (tab_line_count >= 3 and wide_space_lines >= 4)
    )

    return {
        "line_count": len(lines),
        "pipe_line_count": len(pipe_lines),
        "pipe_token_total": pipe_token_total,
        "wide_space_lines": wide_space_lines,
        "tab_line_count": tab_line_count,
        "markdown_table_lines": markdown_table_lines,
        "consistent_pipe_pattern": consistent_pipe_pattern,
        "probable_table": probable_table,
    }


def _has_strong_layout_evidence(layout_profile: dict[str, Any]) -> bool:
    signals = {
        _safe_str(signal, max_len=80)
        for signal in (layout_profile.get("signals") or [])
        if _safe_str(signal, max_len=80)
    }
    confidence = _clamp_float(layout_profile.get("confidence"), default=0.0, min_value=0.0, max_value=1.0)
    table_count = _clamp_int(layout_profile.get("table_count"), default=0, min_value=0, max_value=200)
    column_count = _clamp_int(layout_profile.get("column_count"), default=1, min_value=1, max_value=4)
    return (
        any(signal in STRONG_LAYOUT_SIGNAL_HINTS for signal in signals)
        or column_count >= 3
        or table_count >= 2
        or confidence >= 0.86
    )


def _effective_detected_layout(layout_profile: dict[str, Any], resume_text: str) -> str:
    detected_layout = _safe_str(layout_profile.get("detected_layout"), max_len=32).lower() or "unknown"
    if detected_layout not in {"single_column", "multi_column", "hybrid", "unknown"}:
        detected_layout = "unknown"
    if detected_layout not in {"multi_column", "hybrid"}:
        return detected_layout

    if _has_strong_layout_evidence(layout_profile):
        return detected_layout

    confidence = _clamp_float(layout_profile.get("confidence"), default=0.0, min_value=0.0, max_value=1.0)
    text_signals = _text_layout_signals(resume_text)
    if text_signals["probable_table"]:
        return "hybrid" if detected_layout == "multi_column" else detected_layout

    if detected_layout == "multi_column" and confidence < 0.72:
        return "unknown"
    if detected_layout == "hybrid" and confidence < 0.62:
        return "unknown"

    # Do not classify as multi/hybrid from weak spacing noise only.
    if (
        text_signals["pipe_line_count"] <= 1
        and text_signals["wide_space_lines"] < 6
        and text_signals["tab_line_count"] < 3
        and text_signals["markdown_table_lines"] == 0
    ):
        return "unknown"

    return detected_layout


def _safe_issue_examples(raw_value: Any, *, max_items: int = 6) -> list[dict[str, str]]:
    if not isinstance(raw_value, list):
        return []
    output: list[dict[str, str]] = []
    for item in raw_value:
        if not isinstance(item, dict):
            continue
        text = _safe_str(item.get("text"), max_len=260)
        reason = _safe_str(item.get("reason"), max_len=220)
        suggestion = _safe_str(item.get("suggestion"), max_len=220)
        severity = _safe_str(item.get("severity"), max_len=12).lower()
        if severity not in {"low", "medium", "high"}:
            severity = "medium"
        if not text or not reason or not suggestion:
            continue
        output.append(
            {
                "text": text,
                "reason": reason,
                "suggestion": suggestion,
                "severity": severity,
            }
        )
        if len(output) >= max_items:
            break
    return output


def _is_section_title_line(line: str) -> bool:
    stripped = line.strip()
    lower = stripped.lower()
    if len(stripped.split()) <= 5 and any(
        token in lower
        for token in {"summary", "experience", "skills", "education", "projects", "certifications"}
    ):
        return True
    return bool(re.fullmatch(r"[A-Z\s/&\-]{3,}", stripped) and len(stripped.split()) <= 6)


def _is_contact_or_header_line(line: str) -> bool:
    lower = line.lower()
    if EMAIL_RE.search(line) or PHONE_RE.search(line):
        return True
    if any(token in lower for token in {"linkedin", "github", "portfolio", "contact", "http://", "https://"}):
        return True
    return False


def _extract_candidate_experience_lines(lines: list[str]) -> list[str]:
    candidates: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if _is_section_title_line(stripped):
            continue
        if index <= 3 and _is_contact_or_header_line(stripped):
            continue
        if len(stripped) < 28:
            continue
        # keep only natural sentence/bullet-like content
        if not re.search(r"[A-Za-z]", stripped):
            continue
        candidates.append(stripped)
    return candidates[:120]


def _count_excluded_numeric_tokens(line: str) -> int:
    tokens = re.findall(r"\b\d+(?:[\.,]\d+)?%?\b", line)
    if not tokens:
        return 0
    lower = line.lower()
    excluded = 0
    for token in tokens:
        plain = token.rstrip("%")
        if re.fullmatch(r"(19|20)\d{2}", plain):
            excluded += 1
            continue
        if re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", line):
            excluded += 1
            continue
        if PHONE_RE.search(line) and len(plain) >= 6:
            excluded += 1
            continue
        if "years" in lower and plain.isdigit() and not token.endswith("%"):
            excluded += 1
    return excluded


def _line_has_impact_quantification(line: str) -> bool:
    lower = line.lower()
    if not re.search(r"\d", line):
        return False
    if _is_contact_or_header_line(line):
        return False
    if re.search(r"\b(?:19|20)\d{2}\s*[-–/]\s*(?:19|20)\d{2}\b", lower):
        # explicit date range alone should not be counted as impact
        if not any(keyword in lower for keyword in QUANT_IMPACT_KEYWORDS):
            return False
    if re.search(r"\d+\s*%", lower):
        return True
    if re.search(r"[$€£]\s*\d", line):
        return True
    if re.search(r"\b\d+(?:\.\d+)?x\b", lower):
        return True
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:ms|s|sec|seconds|minutes|hours|days|weeks|months)\b", lower):
        if any(verb in lower for verb in IMPACT_VERB_HINTS):
            return True
    if any(keyword in lower for keyword in QUANT_IMPACT_KEYWORDS):
        if re.search(r"\b\d+(?:[\.,]\d+)?\b", line) and any(verb in lower for verb in IMPACT_VERB_HINTS):
            return True
    if re.search(r"\b\d+\s*(?:users|requests|transactions|incidents|tickets|deployments|pipelines)\b", lower):
        return True
    return False


def _analyze_quantifying_impact(lines: list[str]) -> dict[str, Any]:
    candidates = _extract_candidate_experience_lines(lines)
    quantified_lines: list[str] = []
    unquantified_lines: list[str] = []
    excluded_numeric_tokens = 0

    for line in candidates:
        excluded_numeric_tokens += _count_excluded_numeric_tokens(line)
        if _line_has_impact_quantification(line):
            quantified_lines.append(line)
        else:
            unquantified_lines.append(line)

    scanned = len(candidates)
    quantified = len(quantified_lines)
    ratio = round((quantified / scanned), 2) if scanned else 0.0
    if scanned == 0:
        issues = 1
        score = 40
    elif ratio >= 0.45:
        issues = 0
        score = _clamp_int(int(round(ratio * 100)), default=80, min_value=0, max_value=100)
    elif ratio >= 0.25:
        issues = 1
        score = _clamp_int(int(round(ratio * 100)), default=55, min_value=0, max_value=100)
    else:
        issues = min(6, max(2, scanned - quantified))
        score = _clamp_int(int(round(ratio * 100)), default=30, min_value=0, max_value=100)

    issue_examples = [
        {
            "text": line,
            "reason": "No measurable impact signal detected in this bullet.",
            "suggestion": "Add one concrete metric (%, time, cost, volume) and outcome context.",
            "severity": "medium" if issues <= 2 else "high",
        }
        for line in unquantified_lines[:3]
    ]
    pass_reasons = (
        [f"{quantified}/{scanned} experience bullets include measurable outcomes.", "Impact metrics are tied to delivery statements."]
        if issues == 0 and scanned > 0
        else []
    )

    return {
        "issues": issues,
        "score": score,
        "evidence": quantified_lines[:3],
        "issue_examples": issue_examples,
        "pass_reasons": pass_reasons,
        "metrics": {
            "experience_bullets_scanned": scanned,
            "quantified_bullets": quantified,
            "quantified_ratio": ratio,
            "excluded_numeric_tokens": excluded_numeric_tokens,
        },
        "rationale": (
            f"Quantified bullets={quantified}/{scanned}, ratio={ratio}, excluded_numeric_tokens={excluded_numeric_tokens}."
            if scanned
            else "Not enough experience bullets were detected for full quantification scoring."
        ),
    }


def _normalize_similarity_text(line: str) -> str:
    tokens = [token for token in _tokenize(line) if token not in STOPWORDS]
    return " ".join(tokens)


def _analyze_repetition(lines: list[str]) -> dict[str, Any]:
    candidates = _extract_candidate_experience_lines(lines)
    normalized = [_normalize_similarity_text(line) for line in candidates]
    norm_counter = Counter(item for item in normalized if item)

    exact_duplicate_groups = sum(1 for _, count in norm_counter.items() if count > 1)
    exact_duplicate_issues = sum((count - 1) for _, count in norm_counter.items() if count > 1)

    issue_examples: list[dict[str, str]] = []
    for norm_line, count in norm_counter.items():
        if count <= 1:
            continue
        sample = next((line for line, norm in zip(candidates, normalized) if norm == norm_line), "")
        issue_examples.append(
            {
                "text": sample,
                "reason": f"Exact duplicate phrasing appears {count} times.",
                "suggestion": "Merge duplicate lines and keep only one strongest outcome bullet.",
                "severity": "high" if count > 2 else "medium",
            }
        )
        if len(issue_examples) >= 3:
            break

    near_duplicate_pairs = 0
    max_compare = min(len(normalized), 40)
    for i in range(max_compare):
        if len(normalized[i]) < 25:
            continue
        for j in range(i + 1, max_compare):
            if len(normalized[j]) < 25 or normalized[i] == normalized[j]:
                continue
            score = SequenceMatcher(None, normalized[i], normalized[j]).ratio()
            if score < 0.84:
                continue
            near_duplicate_pairs += 1
            if len(issue_examples) < 5:
                issue_examples.append(
                    {
                        "text": f"{candidates[i]} | {candidates[j]}",
                        "reason": f"Near-duplicate bullet pair detected (similarity {score:.2f}).",
                        "suggestion": "Differentiate one bullet with specific context or measurable outcome.",
                        "severity": "medium",
                    }
                )

    bullet_lines = [line for line in candidates if line.startswith(("-", "*", "•"))]
    starter_counter: Counter[str] = Counter()
    for line in bullet_lines:
        words = [token for token in _tokenize(line) if token]
        if words:
            starter_counter[words[0]] += 1
    dominant_starter_ratio = (
        max(starter_counter.values()) / max(len(bullet_lines), 1)
        if starter_counter and bullet_lines
        else 0.0
    )
    starter_issue = 1 if len(bullet_lines) >= 4 and dominant_starter_ratio >= 0.6 else 0
    if starter_issue and len(issue_examples) < 6:
        dominant = starter_counter.most_common(1)[0][0] if starter_counter else "same"
        issue_examples.append(
            {
                "text": f"Frequent bullet starter: '{dominant}'",
                "reason": "Many bullets start with the same verb pattern.",
                "suggestion": "Vary bullet starters to improve readability and reduce repetition signal.",
                "severity": "low",
            }
        )

    issues = _clamp_int(
        exact_duplicate_issues + near_duplicate_pairs + starter_issue,
        default=0,
        min_value=0,
        max_value=10,
    )
    score = _clamp_int(100 - (issues * 16), default=100, min_value=0, max_value=100)
    pass_reasons = (
        [
            "No duplicate or near-duplicate bullet patterns detected.",
            f"Dominant starter ratio is {dominant_starter_ratio:.2f}, which is within a healthy range.",
        ]
        if issues == 0
        else []
    )
    evidence = [item["text"] for item in issue_examples[:3]] if issues > 0 else candidates[:3]
    return {
        "issues": issues,
        "score": score,
        "evidence": evidence,
        "issue_examples": issue_examples[:6],
        "pass_reasons": pass_reasons,
        "metrics": {
            "exact_duplicate_groups": exact_duplicate_groups,
            "near_duplicate_pairs": near_duplicate_pairs,
            "dominant_starter_ratio": round(dominant_starter_ratio, 2),
        },
        "rationale": (
            f"Exact duplicate groups={exact_duplicate_groups}, near duplicate pairs={near_duplicate_pairs}, "
            f"dominant starter ratio={dominant_starter_ratio:.2f}."
        ),
    }


def _deterministic_spelling_candidates(lines: list[str]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for line in lines[:120]:
        stripped = line.strip()
        if not stripped or len(stripped) < 4:
            continue

        if re.search(r"\s{2,}", stripped):
            candidates.append(
                {
                    "text": stripped,
                    "reason": "Contains repeated spacing.",
                    "suggestion": "Normalize spacing to single spaces.",
                    "severity": "low",
                }
            )
        if re.search(r"[!?.,]{2,}", stripped):
            candidates.append(
                {
                    "text": stripped,
                    "reason": "Contains repeated punctuation.",
                    "suggestion": "Use a single punctuation mark for sentence endings.",
                    "severity": "medium",
                }
            )
        if re.match(r"^[a-z]", stripped) and len(stripped.split()) >= 4:
            candidates.append(
                {
                    "text": stripped,
                    "reason": "Sentence starts with lowercase where capitalization is expected.",
                    "suggestion": "Capitalize sentence start and proper nouns.",
                    "severity": "low",
                }
            )
        lower = stripped.lower()
        for typo, correction in COMMON_TYPO_MAP.items():
            if re.search(rf"\b{re.escape(typo)}\b", lower):
                candidates.append(
                    {
                        "text": stripped,
                        "reason": f"Possible misspelling detected: '{typo}'.",
                        "suggestion": f"Replace '{typo}' with '{correction}'.",
                        "severity": "high",
                    }
                )
        if re.search(r"\bi\b", stripped):
            candidates.append(
                {
                    "text": stripped,
                    "reason": "Standalone lowercase 'i' detected.",
                    "suggestion": "Use uppercase 'I' in English text.",
                    "severity": "low",
                }
            )

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in candidates:
        key = (item["text"], item["reason"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= 10:
            break
    return deduped


def _analyze_spelling_grammar(*, locale: str, lines: list[str]) -> dict[str, Any]:
    candidates = _deterministic_spelling_candidates(lines)
    validated_issues = candidates
    validation_mode = "deterministic"

    llm_payload: dict[str, Any] | None = None
    if _strict_llm_required():
        llm_payload = json_completion_required(
            system_prompt=(
                "You are a resume proofreading assistant. Validate only real language issues and return strict JSON."
            ),
            user_prompt=(
                f"Language: {_locale_language_name(locale)}.\n"
                "Validate spelling/grammar candidates and refine fixes.\n"
                "Return JSON schema:\n"
                "{"
                "\"issues\":[{\"text\":\"...\",\"reason\":\"...\",\"suggestion\":\"...\",\"severity\":\"low|medium|high\"}]"
                "}\n"
                "Keep 0-8 issues only. Remove false positives.\n\n"
                f"Candidate issues: {candidates}\n"
                f"Resume lines: {lines[:35]}\n"
            ),
            temperature=0.1,
            max_output_tokens=700,
        )
    elif candidates:
        llm_payload = json_completion(
            system_prompt=(
                "You are a resume proofreading assistant. Validate only real language issues and return strict JSON."
            ),
            user_prompt=(
                f"Language: {_locale_language_name(locale)}.\n"
                "Validate spelling/grammar candidates and refine fixes.\n"
                "Return JSON schema:\n"
                "{"
                "\"issues\":[{\"text\":\"...\",\"reason\":\"...\",\"suggestion\":\"...\",\"severity\":\"low|medium|high\"}]"
                "}\n"
                "Keep 0-8 issues only. Remove false positives.\n\n"
                f"Candidate issues: {candidates}\n"
                f"Resume lines: {lines[:35]}\n"
            ),
            temperature=0.1,
            max_output_tokens=700,
        )
    if llm_payload:
        parsed = _safe_issue_examples(llm_payload.get("issues"), max_items=8)
        if parsed:
            validated_issues = parsed
            validation_mode = "llm"
        elif _strict_llm_required():
            raw_issues = llm_payload.get("issues")
            if isinstance(raw_issues, list) and len(raw_issues) == 0:
                validated_issues = []
                validation_mode = "llm"
            else:
                raise QualityEnforcementError(
                    "AI quality mode requires validated spelling/grammar issue details, but validation output was invalid.",
                    status_code=503,
                )

    issues = _clamp_int(len(validated_issues), default=0, min_value=0, max_value=15)
    score = _clamp_int(100 - (issues * 12), default=100, min_value=0, max_value=100)
    pass_reasons = (
        [
            "No spelling or grammar anomalies were detected in scanned resume lines.",
            f"Validation mode: {validation_mode}.",
        ]
        if issues == 0
        else []
    )

    return {
        "issues": issues,
        "score": score,
        "evidence": [item["text"] for item in validated_issues[:3]] if issues > 0 else lines[:2],
        "issue_examples": validated_issues,
        "pass_reasons": pass_reasons,
        "metrics": {
            "sentences_scanned": len([line for line in lines if len(line.split()) >= 3]),
            "candidates_found": len(candidates),
            "validated_issues": issues,
            "validation_mode": validation_mode,
        },
        "rationale": (
            f"Grammar validation mode={validation_mode}, candidates={len(candidates)}, validated_issues={issues}."
        ),
    }


DATE_RE = re.compile(r"\b(?:20\d{2}|19\d{2})\b|\b\d{1,2}[/\-]\d{2,4}\b")

SECTION_KEYWORDS: dict[str, set[str]] = {
    "summary": {"summary", "professional profile", "profile", "objective", "about"},
    "experience": {"experience", "employment", "work history", "professional experience"},
    "skills": {"skills", "competencies", "tech stack", "technologies", "technical skills"},
    "education": {"education", "degree", "university", "bachelor", "master", "academic"},
    "certifications": {"certifications", "certification", "certificates", "accreditations"},
    "projects": {"projects", "portfolio", "personal projects"},
}

PREDICTED_SKILL_MAP: dict[str, list[str]] = {
    "python": ["django", "flask", "fastapi", "pandas", "numpy"],
    "react": ["redux", "next.js", "typescript", "webpack"],
    "node": ["express", "nest.js", "typescript", "mongodb"],
    "node.js": ["express", "nest.js", "typescript"],
    "docker": ["kubernetes", "ci/cd", "terraform", "helm"],
    "aws": ["ec2", "s3", "lambda", "cloudformation", "terraform"],
    "azure": ["devops", "functions", "cosmos", "terraform"],
    "java": ["spring", "spring boot", "maven", "gradle", "hibernate"],
    "c#": [".net", "entity framework", "azure", "blazor"],
    "sql": ["postgresql", "mysql", "database", "orm"],
    "kubernetes": ["docker", "helm", "terraform", "ci/cd"],
    "typescript": ["react", "next.js", "node.js", "angular"],
    "angular": ["rxjs", "ngrx", "typescript"],
    "postgresql": ["sql", "database", "orm", "redis"],
    "mongodb": ["mongoose", "nosql", "redis"],
    "redis": ["caching", "message queue", "celery"],
    "graphql": ["apollo", "rest", "api"],
    "tensorflow": ["pytorch", "machine learning", "python", "numpy"],
    "pytorch": ["tensorflow", "machine learning", "python"],
}


def _build_skills_comparison(
    resume_text: str, jd_text: str, matched_terms: list[str], missing_terms: list[str],
) -> dict[str, Any]:
    resume_tokens = _tokenize(resume_text)
    jd_tokens = _tokenize(jd_text)
    resume_counts: Counter[str] = Counter(resume_tokens)
    jd_counts: Counter[str] = Counter(jd_tokens)

    all_terms = set(matched_terms) | set(missing_terms)
    hard_skill_set = TOOL_TERMS | ROLE_SIGNAL_TERMS | DOMAIN_TERMS
    matched_set = set(matched_terms)

    hard_skills: list[dict[str, Any]] = []
    soft_skills: list[dict[str, Any]] = []

    for term in sorted(all_terms):
        item = {
            "term": term,
            "jd_count": jd_counts.get(term, 0),
            "resume_count": resume_counts.get(term, 0),
            "matched": term in matched_set,
        }
        if term in SOFT_SKILL_TERMS:
            soft_skills.append(item)
        elif term in hard_skill_set or item["jd_count"] > 0:
            hard_skills.append(item)

    jd_soft_terms = [t for t in jd_tokens if t in SOFT_SKILL_TERMS]
    for term in sorted(set(jd_soft_terms)):
        if not any(s["term"] == term for s in soft_skills):
            soft_skills.append({
                "term": term,
                "jd_count": jd_counts.get(term, 0),
                "resume_count": resume_counts.get(term, 0),
                "matched": resume_counts.get(term, 0) > 0,
            })

    hard_skills.sort(key=lambda x: (not x["matched"], -x["jd_count"]))
    soft_skills.sort(key=lambda x: (not x["matched"], -x["jd_count"]))

    hard_matched = sum(1 for s in hard_skills if s["matched"])
    soft_matched = sum(1 for s in soft_skills if s["matched"])

    predicted: list[str] = []
    matched_lower = {t.lower() for t in matched_terms}
    jd_lower_set = {t.lower() for t in jd_tokens}
    for term in matched_terms[:10]:
        for candidate in PREDICTED_SKILL_MAP.get(term.lower(), []):
            if candidate not in jd_lower_set and candidate not in matched_lower and candidate not in predicted:
                predicted.append(candidate)
            if len(predicted) >= 6:
                break
        if len(predicted) >= 6:
            break

    return {
        "hard_skills": hard_skills,
        "soft_skills": soft_skills,
        "hard_matched": hard_matched,
        "hard_total": len(hard_skills),
        "soft_matched": soft_matched,
        "soft_total": len(soft_skills),
        "predicted_skills": predicted,
    }


def _build_searchability(resume_text: str) -> dict[str, Any]:
    lines = [line.strip() for line in resume_text.splitlines()]
    non_empty_lines = [line for line in lines if line]
    name = non_empty_lines[0] if non_empty_lines else ""

    email_match = EMAIL_RE.search(resume_text)
    phone_match = PHONE_RE.search(resume_text)
    email = email_match.group(0) if email_match else ""
    phone = phone_match.group(0) if phone_match else ""

    resume_lower = resume_text.lower()
    detected: list[str] = []
    missing: list[str] = []
    for section_name, keywords in SECTION_KEYWORDS.items():
        if any(kw in resume_lower for kw in keywords):
            detected.append(section_name)
        else:
            missing.append(section_name)

    date_count = len(DATE_RE.findall(resume_text))
    word_count = len(resume_text.split())

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "has_email": bool(email),
        "has_phone": bool(phone),
        "sections_detected": detected,
        "sections_missing": missing,
        "date_formats_found": date_count,
        "word_count": word_count,
        "line_count": len(non_empty_lines),
    }


def _build_recruiter_tips(
    resume_text: str, jd_text: str, years_required: int, seniority: str,
) -> dict[str, Any]:
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()
    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]

    first_two_lines = " ".join(lines[:2]).lower() if lines else ""
    jd_title_terms = [
        t for t in _important_terms(jd_text, limit=10)
        if len(t) > 3 and t not in STOPWORDS and t not in LOW_SIGNAL_TERMS
    ][:5]
    title_found = first_two_lines if lines else ""
    title_expected = " ".join(jd_title_terms[:3])
    title_match = any(t in first_two_lines for t in jd_title_terms) if jd_title_terms else False

    resume_years = _seniority_to_years(seniority)
    years_match = years_required <= 0 or resume_years >= years_required

    degree_keywords = {"bachelor", "master", "phd", "degree", "bsc", "msc", "mba"}
    resume_has_degree = any(kw in resume_lower for kw in degree_keywords)
    jd_requires_degree = any(kw in jd_lower for kw in degree_keywords)
    education_match = not jd_requires_degree or resume_has_degree

    measurable_lines = [
        line for line in lines
        if re.search(r"\d+\s*%|\$\s*\d|[0-9]+x\b|\b\d{2,}\b", line)
    ]
    measurable_count = len(measurable_lines)
    if measurable_count >= 4:
        measurable_status = "good"
    elif measurable_count >= 2:
        measurable_status = "needs_work"
    else:
        measurable_status = "missing"

    word_count = len(resume_text.split())
    if 400 <= word_count <= 800:
        wc_status = "good"
    elif word_count < 400:
        wc_status = "short"
    else:
        wc_status = "long"

    return {
        "job_title_match": {"found": title_found[:80], "expected": title_expected, "match": title_match},
        "years_match": {"resume_years": resume_years, "jd_required": years_required, "match": years_match},
        "education_match": {"resume_has_degree": resume_has_degree, "jd_requires_degree": jd_requires_degree, "match": education_match},
        "measurable_results": {"count": measurable_count, "status": measurable_status},
        "word_count": {"count": word_count, "status": wc_status},
    }


def _build_ats_report(
    *,
    payload: ToolRequest,
    base: dict[str, Any],
    parsing_flags: list[str],
    credibility: dict[str, Any],
    lines: list[str],
) -> dict[str, Any]:
    resume = payload.resume_text
    resume_lower = resume.lower()
    jd_lower = payload.job_description_text.lower()
    layout_profile = base.get("layout_profile") if isinstance(base.get("layout_profile"), dict) else _coerce_layout_profile(payload.resume_layout_profile, resume)
    effective_layout = _effective_detected_layout(layout_profile, resume)
    strong_layout = _has_strong_layout_evidence(layout_profile)
    layout_fit = base.get("layout_fit_for_target") if isinstance(base.get("layout_fit_for_target"), dict) else _layout_fit_for_target(
        layout_profile=layout_profile,
        target_region=payload.candidate_profile.target_region,
        jd_text=payload.job_description_text,
        resume_text=resume,
    )
    file_meta = base.get("resume_file_meta") if isinstance(base.get("resume_file_meta"), dict) else _coerce_resume_file_meta(payload.resume_file_meta)
    parsing_penalty = _parsing_penalty(resume, layout_profile=layout_profile, layout_fit=layout_fit)

    parse_rate_issues = 0 if parsing_penalty < 12 else 1 if parsing_penalty < 24 else 2
    quantifying_analysis = _analyze_quantifying_impact(lines)
    quantifying_issues = _clamp_int(quantifying_analysis.get("issues"), default=0, min_value=0, max_value=10)
    repetition_analysis = _analyze_repetition(lines)
    repetition_issues = _clamp_int(repetition_analysis.get("issues"), default=0, min_value=0, max_value=10)
    spelling_analysis = _analyze_spelling_grammar(locale=payload.locale, lines=lines)
    spelling_issues = _clamp_int(spelling_analysis.get("issues"), default=0, min_value=0, max_value=15)

    has_summary = any(keyword in resume_lower for keyword in {"summary", "professional profile", "profile"})
    has_experience = any(keyword in resume_lower for keyword in {"experience", "employment"})
    has_skills = any(keyword in resume_lower for keyword in {"skills", "competencies", "tech stack"})
    has_education = any(keyword in resume_lower for keyword in {"education", "degree", "university", "bachelor", "master"})
    essential_missing = sum(1 for present in [has_summary, has_experience, has_skills, has_education] if not present)

    contact_issues = 0
    if not EMAIL_RE.search(resume):
        contact_issues += 1
    if not PHONE_RE.search(resume):
        contact_issues += 1

    email_issue = 0 if EMAIL_RE.search(resume) else 1
    header_lines = [line.lower() for line in lines[:3]]
    header_has_hyperlink = any("http://" in line or "https://" in line or "linkedin.com" in line for line in header_lines)
    header_density = _clamp_float(layout_profile.get("header_link_density"), default=0.0, min_value=0.0, max_value=1.0)
    header_has_hyperlink = header_has_hyperlink or header_density >= 0.5
    hyperlink_header_issue = 1 if header_has_hyperlink else 0
    design_issues = 0 if parsing_penalty < 12 else 1 if parsing_penalty < 24 else 2
    detected_layout = effective_layout.replace("_", " ")
    layout_note = _safe_str(layout_fit.get("format_recommendation"), max_len=220)
    display_column_count = _clamp_int(layout_profile.get("column_count"), default=1, min_value=1, max_value=4)
    if effective_layout in {"single_column", "unknown"} and not strong_layout:
        display_column_count = 1
    extension = _safe_str(file_meta.get("extension"), max_len=16).lower()
    if extension == "doc":
        file_format_issues = 2
    elif extension in {"pdf", "docx"}:
        file_format_issues = 0
    elif extension in {"txt", "md", "rtf"}:
        file_format_issues = 1
    else:
        file_format_issues = 0 if not extension else 1

    matched_terms = set(base["matched_terms"])
    missing_terms = set(base["missing_terms"])
    hard_terms = sorted({term for term in matched_terms.union(missing_terms) if term in TOOL_TERMS or term in ROLE_SIGNAL_TERMS})
    hard_matched = sum(1 for term in hard_terms if term in matched_terms)
    hard_issues = 0 if not hard_terms else _clamp_int(len(hard_terms) - hard_matched, default=0, min_value=0, max_value=8)

    soft_terms = sorted({term for term in SOFT_SKILL_TERMS if term in jd_lower})
    soft_matched = sum(1 for term in soft_terms if term in resume_lower)
    soft_issues = 0 if not soft_terms else _clamp_int(len(soft_terms) - soft_matched, default=0, min_value=0, max_value=6)

    action_bullets = _clamp_int(credibility.get("action_bullets"), default=0, min_value=0, max_value=120)
    action_issues = 0 if action_bullets >= 6 else 1 if action_bullets >= 3 else 2

    word_count = len(_tokenize(resume))
    if word_count < 220 or word_count > 1200:
        resume_length_issues = 2
        resume_length_score = 48
    elif word_count < 320 or word_count > 980:
        resume_length_issues = 1
        resume_length_score = 72
    else:
        resume_length_issues = 0
        resume_length_score = 92

    bullet_candidates = [line.strip(" -*\u2022\t") for line in lines if line.strip().startswith(("-", "*", "\u2022"))]
    if not bullet_candidates:
        bullet_candidates = [line.strip() for line in lines if len(_tokenize(line)) >= 8][:24]
    long_bullets = [bullet for bullet in bullet_candidates if len(_tokenize(bullet)) > 32]
    long_bullet_issues = _clamp_int(len(long_bullets), default=0, min_value=0, max_value=8)
    long_bullet_score = _clamp_int(100 - (long_bullet_issues * 14), default=86, min_value=20, max_value=100)

    personality_terms = {
        "leadership",
        "led",
        "owned",
        "ownership",
        "mentored",
        "coached",
        "collaborated",
        "communication",
        "cross-functional",
        "stakeholder",
        "initiative",
    }
    personality_hits = sorted([term for term in personality_terms if term in resume_lower])
    personality_issues = 0 if len(personality_hits) >= 2 else 1
    personality_score = 92 if personality_issues == 0 else 64

    active_voice_issues = 0 if action_bullets >= 6 else 1 if action_bullets >= 3 else 2
    active_voice_score = _clamp_int(100 - (active_voice_issues * 26), default=74, min_value=30, max_value=100)

    humanization = _humanization_report(resume)
    buzzword_count = _clamp_int(humanization.get("cliche_count"), default=0, min_value=0, max_value=10)
    buzzword_issues = buzzword_count
    buzzword_score = _clamp_int(100 - (buzzword_count * 12), default=84, min_value=20, max_value=100)

    first_line = lines[0].lower() if lines else ""
    title_terms = [term for term in _important_terms(payload.job_description_text, limit=12) if len(term) > 3 and term not in STOPWORDS][:5]
    tailored_title_hit = any(term in first_line for term in title_terms)
    tailored_title_issue = 0 if tailored_title_hit else 1

    header_lines = lines[:3]

    def make_check(
        check_id: str,
        label: str,
        issues: int,
        description: str,
        recommendation: str,
        score: int | None = None,
        evidence: list[str] | None = None,
        rationale: str = "",
        issue_examples: list[dict[str, str]] | None = None,
        pass_reasons: list[str] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_score = score if score is not None else max(0, min(100, 100 - (issues * 22)))
        resolved_evidence = [item for item in (evidence or []) if item][:4]
        resolved_issue_examples = _safe_issue_examples(issue_examples or [], max_items=6)
        resolved_pass_reasons = _safe_str_list(pass_reasons or [], max_items=4, max_len=220)
        resolved_rationale = _safe_str(rationale, max_len=260)
        if issues > 0 and not resolved_issue_examples:
            fallback_text = resolved_evidence[0] if resolved_evidence else "Insufficient direct snippet available."
            resolved_issue_examples = [
                {
                    "text": fallback_text,
                    "reason": "Issue detected by ATS heuristic evaluation.",
                    "suggestion": recommendation,
                    "severity": "medium",
                }
            ]
        if issues <= 0 and not resolved_pass_reasons:
            resolved_pass_reasons = [
                "No issues detected for this check based on current resume evidence.",
                f"Check score is {resolved_score}%.",
            ]
        return {
            "id": check_id,
            "label": label,
            "status": _check_status(issues),
            "issues": issues,
            "issue_label": _issue_label(issues),
            "score": resolved_score,
            "description": description,
            "recommendation": recommendation,
            "evidence": resolved_evidence,
            "rationale": resolved_rationale,
            "issue_examples": resolved_issue_examples,
            "pass_reasons": resolved_pass_reasons,
            "metrics": metrics or {},
        }

    content_checks = [
        make_check(
            "ats_parse_rate",
            "ATS Parse Rate",
            parse_rate_issues,
            f"How reliably ATS can read your resume structure and fields. Detected layout: {detected_layout}.",
            layout_note or "Use one-column layout and avoid complex formatting blocks.",
            score=max(0, min(100, 100 - parsing_penalty)),
            evidence=parsing_flags[:3] or header_lines,
            rationale=f"Parsing penalty={parsing_penalty}, layout={detected_layout}.",
            pass_reasons=(
                ["Layout and formatting patterns are within ATS-safe range."]
                if parse_rate_issues == 0
                else []
            ),
            metrics={
                "parsing_penalty": parsing_penalty,
                "layout_type": detected_layout,
                "layout_fit": _safe_str(layout_fit.get("fit_level"), max_len=20),
            },
        ),
        make_check(
            "repetition",
            "Repetition of Words and Phrases",
            repetition_issues,
            "Detects exact and near-duplicate bullets plus repetitive bullet starters.",
            "Vary verbs, merge duplicate bullets, and keep each bullet focused on one unique outcome.",
            score=_clamp_int(repetition_analysis.get("score"), default=70, min_value=0, max_value=100),
            evidence=[
                _safe_str(item, max_len=240)
                for item in repetition_analysis.get("evidence", [])
                if _safe_str(item, max_len=240)
            ],
            rationale=_safe_str(repetition_analysis.get("rationale"), max_len=260),
            issue_examples=repetition_analysis.get("issue_examples"),
            pass_reasons=repetition_analysis.get("pass_reasons"),
            metrics=repetition_analysis.get("metrics"),
        ),
        make_check(
            "spelling_grammar",
            "Spelling and Grammar",
            spelling_issues,
            "Flags likely grammar, punctuation, and typo quality issues with line-level evidence.",
            "Run a grammar pass and fix punctuation, capitalization, and typo signals in flagged lines.",
            score=_clamp_int(spelling_analysis.get("score"), default=80, min_value=0, max_value=100),
            evidence=[
                _safe_str(item, max_len=240)
                for item in spelling_analysis.get("evidence", [])
                if _safe_str(item, max_len=240)
            ],
            rationale=_safe_str(spelling_analysis.get("rationale"), max_len=260),
            issue_examples=spelling_analysis.get("issue_examples"),
            pass_reasons=spelling_analysis.get("pass_reasons"),
            metrics=spelling_analysis.get("metrics"),
        ),
        make_check(
            "quantifying_impact",
            "Quantifying Impact",
            quantifying_issues,
            "Checks if experience bullets are backed by measurable outcomes and examples.",
            "Add metrics (%, time saved, revenue, latency, volume) in key bullets.",
            score=_clamp_int(quantifying_analysis.get("score"), default=55, min_value=0, max_value=100),
            evidence=[
                _safe_str(item, max_len=240)
                for item in quantifying_analysis.get("evidence", [])
                if _safe_str(item, max_len=240)
            ],
            rationale=_safe_str(quantifying_analysis.get("rationale"), max_len=260),
            issue_examples=quantifying_analysis.get("issue_examples"),
            pass_reasons=quantifying_analysis.get("pass_reasons"),
            metrics=quantifying_analysis.get("metrics"),
        ),
    ]

    resume_sections_checks = [
        make_check(
            "contact_information",
            "Contact Information",
            contact_issues,
            "Checks if recruiter-contact essentials are present and parseable.",
            "Keep email and phone plain text in the top section.",
            evidence=header_lines,
            rationale=f"Contact issues={contact_issues}.",
            issue_examples=(
                []
                if contact_issues == 0
                else [
                    {
                        "text": "Contact block",
                        "reason": "Email or phone is missing or hard to parse.",
                        "suggestion": "Add plain-text email and international-format phone in the header.",
                        "severity": "high" if contact_issues > 1 else "medium",
                    }
                ]
            ),
            pass_reasons=(
                ["Email and phone were detected in parseable plain text."]
                if contact_issues == 0
                else []
            ),
            metrics={
                "email_found": bool(EMAIL_RE.search(resume)),
                "phone_found": bool(PHONE_RE.search(resume)),
                "contact_issues": contact_issues,
            },
        ),
        make_check(
            "essential_sections",
            "Essential Sections",
            essential_missing,
            "Checks if core sections exist (Summary, Experience, Skills, Education).",
            "Ensure all core sections are present and clearly labeled.",
            evidence=[f"Summary={has_summary}", f"Experience={has_experience}", f"Skills={has_skills}", f"Education={has_education}"],
            rationale=f"Missing essential section count={essential_missing}.",
            pass_reasons=(
                ["All core sections (Summary, Experience, Skills, Education) were detected."]
                if essential_missing == 0
                else []
            ),
            metrics={
                "summary_present": has_summary,
                "experience_present": has_experience,
                "skills_present": has_skills,
                "education_present": has_education,
                "missing_sections": essential_missing,
            },
        ),
        make_check(
            "personality_showcase",
            "Personality Showcase",
            personality_issues,
            "Checks if your resume reflects people-impact signals such as leadership, collaboration, and ownership.",
            "Include one or two bullets that show mentoring, stakeholder communication, or cross-functional ownership.",
            score=personality_score,
            evidence=personality_hits[:6],
            rationale=f"Personality signal hits={len(personality_hits)}.",
            issue_examples=(
                []
                if personality_issues == 0
                else [
                    {
                        "text": "Resume narrative",
                        "reason": "Limited evidence of interpersonal impact and collaboration signals.",
                        "suggestion": "Add outcome-backed bullets showing leadership, mentoring, or stakeholder alignment.",
                        "severity": "medium",
                    }
                ]
            ),
            pass_reasons=(
                ["Resume shows clear people-impact signals (leadership/collaboration/ownership)."]
                if personality_issues == 0
                else []
            ),
            metrics={"personality_signals_found": len(personality_hits)},
        ),
    ]

    format_checks = [
        make_check(
            "file_format_size",
            "File Format and Size",
            file_format_issues,
            "Checks whether uploaded format is ATS-friendly for parsing reliability.",
            "Prefer PDF or DOCX for ATS-heavy screening. Avoid legacy .doc.",
            evidence=[f"extension={extension or 'unknown'}", f"source_type={file_meta.get('source_type', 'unknown')}"],
            rationale=f"Format issue score derived from extension='{extension or 'unknown'}'.",
            issue_examples=(
                []
                if file_format_issues == 0
                else [
                    {
                        "text": extension or "unknown",
                        "reason": "File format can reduce ATS parsing reliability.",
                        "suggestion": "Upload as PDF or DOCX and keep file size moderate.",
                        "severity": "high" if extension == "doc" else "medium",
                    }
                ]
            ),
            pass_reasons=(
                ["File format is ATS-friendly for most screening pipelines."]
                if file_format_issues == 0
                else []
            ),
            metrics={
                "extension": extension or "unknown",
                "source_type": _safe_str(file_meta.get("source_type"), max_len=20) or "unknown",
                "format_issues": file_format_issues,
            },
        ),
        make_check(
            "resume_length",
            "Resume Length",
            resume_length_issues,
            "Evaluates whether resume length is balanced for ATS and recruiter scanability.",
            "Keep the document concise and prioritize high-impact evidence in the first page.",
            score=resume_length_score,
            evidence=[f"word_count={word_count}", f"line_count={len(lines)}"],
            rationale=f"Resume length scored from word_count={word_count}.",
            issue_examples=(
                []
                if resume_length_issues == 0
                else [
                    {
                        "text": f"{word_count} words detected",
                        "reason": "Resume length may be too short or too long for fast recruiter screening.",
                        "suggestion": "Target a concise range by trimming low-impact lines or adding missing evidence bullets.",
                        "severity": "high" if resume_length_issues >= 2 else "medium",
                    }
                ]
            ),
            pass_reasons=(["Resume length is within a strong range for ATS and recruiter review."] if resume_length_issues == 0 else []),
            metrics={"word_count": word_count, "line_count": len(lines)},
        ),
        make_check(
            "long_bullet_points",
            "Long Bullet Points",
            long_bullet_issues,
            "Flags bullets that are too long and harder to scan quickly.",
            "Split long bullets into shorter action-impact lines and keep one idea per bullet.",
            score=long_bullet_score,
            evidence=long_bullets[:4],
            rationale=f"Long bullets detected={long_bullet_issues}.",
            issue_examples=(
                []
                if long_bullet_issues == 0
                else [
                    {
                        "text": bullet,
                        "reason": "Bullet length reduces readability and weakens impact clarity.",
                        "suggestion": "Shorten to 16-26 words and keep metric + outcome explicit.",
                        "severity": "medium",
                    }
                    for bullet in long_bullets[:4]
                ]
            ),
            pass_reasons=(["Bullets are concise and readable for fast recruiter review."] if long_bullet_issues == 0 else []),
            metrics={"bullets_scanned": len(bullet_candidates), "long_bullets": long_bullet_issues},
        ),
    ]

    skills_suggestion_checks = [
        make_check(
            "hard_skills",
            "Hard Skills",
            hard_issues,
            "Measures role-critical technical term coverage from the JD.",
            "Add missing hard skills only where you have real evidence.",
            score=max(0, min(100, int(round((hard_matched / max(len(hard_terms), 1)) * 100)))),
            evidence=sorted(list(matched_terms))[:5],
            rationale=f"Matched hard terms={hard_matched}/{max(len(hard_terms), 1)}.",
            issue_examples=(
                []
                if hard_issues == 0
                else [
                    {
                        "text": term,
                        "reason": "Required hard-skill term is missing from resume evidence.",
                        "suggestion": "Add this skill only where it is actually used in your experience bullets.",
                        "severity": "medium",
                    }
                    for term in sorted(list(missing_terms.intersection(set(hard_terms))))[:3]
                ]
            ),
            pass_reasons=(
                [f"Matched {hard_matched} out of {max(len(hard_terms), 1)} role-critical hard skill terms."]
                if hard_issues == 0
                else []
            ),
            metrics={"hard_terms_total": len(hard_terms), "hard_terms_matched": hard_matched},
        ),
        make_check(
            "soft_skills",
            "Soft Skills",
            soft_issues,
            "Measures soft-skill alignment for collaboration and communication signals.",
            "Reflect soft skills through outcomes and responsibilities, not buzzwords.",
            score=max(0, min(100, int(round((soft_matched / max(len(soft_terms), 1)) * 100)) if soft_terms else 100)),
            evidence=[term for term in soft_terms if term in resume_lower][:5],
            rationale=f"Matched soft terms={soft_matched}/{max(len(soft_terms), 1)}.",
            issue_examples=(
                []
                if soft_issues == 0
                else [
                    {
                        "text": term,
                        "reason": "Soft-skill signal appears in JD but is weak in resume evidence.",
                        "suggestion": "Show this skill through a concrete project outcome or collaboration example.",
                        "severity": "low",
                    }
                    for term in soft_terms
                    if term not in resume_lower
                ][:3]
            ),
            pass_reasons=(
                [f"Matched {soft_matched} out of {max(len(soft_terms), 1)} JD soft-skill signals."]
                if soft_issues == 0
                else []
            ),
            metrics={"soft_terms_total": len(soft_terms), "soft_terms_matched": soft_matched},
        ),
    ]

    style_checks = [
        make_check(
            "design",
            "Resume Design",
            design_issues,
            f"Checks layout complexity ({detected_layout}) that can reduce parse reliability.",
            layout_note or "Avoid multi-column blocks, tables, and dense header/footer content.",
            score=max(0, min(100, 100 - parsing_penalty)),
            evidence=[
                f"columns={display_column_count}",
                f"tables={layout_profile.get('table_count', 0)}",
                f"complexity={layout_profile.get('complexity_score', 0)}",
            ],
            rationale=f"Design issues are tied to layout profile and parsing penalty ({parsing_penalty}).",
            pass_reasons=(
                ["Layout complexity is within ATS-safe design thresholds."]
                if design_issues == 0
                else []
            ),
            metrics={
                "column_count": display_column_count,
                "table_count": _clamp_int(layout_profile.get("table_count"), default=0, min_value=0, max_value=200),
                "complexity_score": _clamp_int(layout_profile.get("complexity_score"), default=20, min_value=0, max_value=100),
                "parsing_penalty": parsing_penalty,
            },
        ),
        make_check(
            "email_address",
            "Email Address",
            email_issue,
            "Validates presence of a parseable professional email address.",
            "Use a professional email in plain text format.",
            evidence=[EMAIL_RE.search(resume).group(0)] if EMAIL_RE.search(resume) else [],
            rationale="Email parse check based on regex match in resume content.",
            issue_examples=(
                []
                if email_issue == 0
                else [
                    {
                        "text": "Email address",
                        "reason": "No parseable email address detected.",
                        "suggestion": "Add one professional email in plain text near the top of resume.",
                        "severity": "high",
                    }
                ]
            ),
            pass_reasons=(["A parseable email address was detected."] if email_issue == 0 else []),
            metrics={"email_found": bool(EMAIL_RE.search(resume))},
        ),
        make_check(
            "active_voice",
            "Usage of Active Voice",
            active_voice_issues,
            "Checks whether bullets are written in direct active voice with strong action verbs.",
            "Rewrite passive bullets into active verb-led statements with measurable impact.",
            score=active_voice_score,
            evidence=lines[:4],
            rationale=f"Action-led bullet signal={action_bullets}.",
            issue_examples=(
                []
                if active_voice_issues == 0
                else [
                    {
                        "text": line,
                        "reason": "Line is likely passive or does not start with a strong action verb.",
                        "suggestion": "Begin with a clear action verb and tie the action to an outcome.",
                        "severity": "medium",
                    }
                    for line in lines[:3]
                ]
            ),
            pass_reasons=(
                [f"Detected {action_bullets} action-led bullets with active wording."]
                if active_voice_issues == 0
                else []
            ),
            metrics={"action_bullets_detected": action_bullets},
        ),
        make_check(
            "buzzwords_cliches",
            "Usage of Buzzwords and Cliches",
            buzzword_issues,
            "Detects overused generic phrases that reduce credibility and sound templated.",
            "Replace buzzwords with specific project evidence and outcomes.",
            score=buzzword_score,
            evidence=[_safe_str(term, max_len=80) for term in humanization.get("detected_cliches", [])][:6],
            rationale=f"Cliche phrases detected={buzzword_count}.",
            issue_examples=(
                []
                if buzzword_issues == 0
                else [
                    {
                        "text": _safe_str(term, max_len=120),
                        "reason": "Generic cliche weakens trust and does not show concrete evidence.",
                        "suggestion": "Replace this phrase with a specific achievement and measurable result.",
                        "severity": "low" if buzzword_issues <= 2 else "medium",
                    }
                    for term in humanization.get("detected_cliches", [])[:5]
                ]
            ),
            pass_reasons=(["No major buzzword/cliche patterns detected in the resume narrative."] if buzzword_issues == 0 else []),
            metrics={"cliche_count": buzzword_count},
        ),
        make_check(
            "hyperlink_in_header",
            "Hyperlink in Header",
            hyperlink_header_issue,
            "Checks if dense hyperlinks in the header can affect ATS extraction and field mapping.",
            "Keep only critical links and avoid crowded header contact lines.",
            evidence=header_lines,
            rationale=f"Header link density={header_density}.",
            issue_examples=(
                []
                if hyperlink_header_issue == 0
                else [
                    {
                        "text": "Header contact line",
                        "reason": "Header contains dense links which can confuse ATS field parsing.",
                        "suggestion": "Keep at most 1-2 essential links (e.g., LinkedIn + portfolio).",
                        "severity": "medium",
                    }
                ]
            ),
            pass_reasons=(
                ["Header link density is low and unlikely to disrupt ATS mapping."]
                if hyperlink_header_issue == 0
                else []
            ),
            metrics={"header_link_density": round(header_density, 2), "header_has_hyperlink": header_has_hyperlink},
        ),
        make_check(
            "tailored_title",
            "Tailored Title",
            tailored_title_issue,
            "Checks whether headline/title matches role language.",
            "Align your headline with the target job title and scope.",
            score=100 if tailored_title_hit else 62,
            evidence=[lines[0]] if lines else [],
            rationale=f"Title terms matched={tailored_title_hit}.",
            issue_examples=(
                []
                if tailored_title_issue == 0
                else [
                    {
                        "text": lines[0] if lines else "No headline detected",
                        "reason": "Headline does not clearly align with target role terms.",
                        "suggestion": "Use a headline matching your target title and domain scope.",
                        "severity": "medium",
                    }
                ]
            ),
            pass_reasons=(["Headline aligns with target job-title language."] if tailored_title_issue == 0 else []),
            metrics={"title_terms_checked": len(title_terms), "title_match": tailored_title_hit},
        ),
    ]

    def category(category_id: str, label: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
        issue_count = sum(int(item["issues"]) for item in checks)
        score = int(round(sum(int(item["score"]) for item in checks) / max(len(checks), 1)))
        return {
            "id": category_id,
            "label": label,
            "score": score,
            "issue_count": issue_count,
            "issue_label": _issue_label(issue_count),
            "checks": checks,
        }

    categories = [
        category("content", "Content", content_checks),
        category("format", "Format", format_checks),
        category("skills_suggestion", "Skills Suggestion", skills_suggestion_checks),
        category("resume_sections", "Resume Sections", resume_sections_checks),
        category("style", "Style", style_checks),
    ]

    total_issues = sum(item["issue_count"] for item in categories)
    parsed_content_score = next(
        (check["score"] for check in content_checks if check.get("id") == "ats_parse_rate"),
        max(0, min(100, 100 - parsing_penalty)),
    )
    issue_impact_score = _clamp_int(100 - (total_issues * 4), default=72, min_value=0, max_value=100)
    overall_score = int(round((parsed_content_score * 0.58) + (issue_impact_score * 0.42)))

    return {
        "overall_score": overall_score,
        "total_issues": total_issues,
        "tier_scores": {
            "parsed_content_score": parsed_content_score,
            "issue_impact_score": issue_impact_score,
        },
        "categories": categories,
        "parsing_flags": parsing_flags,
        "layout_profile": layout_profile,
        "layout_fit_for_target": layout_fit,
        "format_recommendation": layout_note,
        "skills_coverage": {
            "hard_terms_total": len(hard_terms),
            "hard_terms_matched": hard_matched,
            "soft_terms_total": len(soft_terms),
            "soft_terms_matched": soft_matched,
        },
    }


def _additional_ai_insights(
    *,
    tool_slug: str,
    locale: str,
    resume_text: str,
    job_description_text: str,
    tool_inputs: dict[str, Any],
    risks: list[RiskItem],
    fix_plan: list[FixPlanItem],
) -> list[str]:
    llm_payload = json_completion(
        system_prompt=(
            "You are a practical career tooling assistant. "
            "Generate concise, evidence-based insights only. "
            "Return strict JSON only."
        ),
        user_prompt=(
            f"Language: {_locale_language_name(locale)}.\n"
            f"Tool slug: {tool_slug}\n"
            "Return JSON schema:\n"
            "{"
            "\"insights\":[\"...\"]"
            "}\n"
            "Rules:\n"
            "- Return 2 to 4 insights\n"
            "- Each insight must be concrete and actionable\n"
            "- Avoid motivational fluff and generic statements\n"
            "- Use only evidence from resume/JD/inputs\n\n"
            f"Tool inputs: {tool_inputs}\n"
            f"Detected risks: {[{'type': risk.type, 'severity': risk.severity, 'message': risk.message} for risk in risks]}\n"
            f"Fix plan: {[{'id': item.id, 'title': item.title, 'impact_score': item.impact_score, 'effort_minutes': item.effort_minutes} for item in fix_plan]}\n"
            f"Resume excerpt:\n{resume_text[:1800]}\n\n"
            f"JD excerpt:\n{job_description_text[:1800]}\n"
        ),
        temperature=0.15,
        max_output_tokens=380,
    )
    if not llm_payload:
        return []
    return _safe_str_list(llm_payload.get("insights"), max_items=4, max_len=240)


def _build_base_analysis(
    payload: ToolRequest,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    _ensure_llm_ready("base-analysis")
    locale = payload.locale
    resume_text = payload.resume_text
    jd_text = payload.job_description_text
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()
    layout_profile = _coerce_layout_profile(payload.resume_layout_profile, resume_text)
    resume_file_meta = _coerce_resume_file_meta(payload.resume_file_meta)
    _emit_progress(
        progress_callback,
        stage="analyzing_experience",
        label="Analyzing your experience",
        percent=42,
        detail="Comparing resume evidence with job requirements and risks.",
    )
    if resume_file_meta["source_type"] == "unknown":
        resume_file_meta["source_type"] = _safe_str(layout_profile.get("source_type"), max_len=20).lower() or "unknown"
    layout_fit = _layout_fit_for_target(
        layout_profile=layout_profile,
        target_region=payload.candidate_profile.target_region,
        jd_text=jd_text,
        resume_text=resume_text,
    )

    jd_terms_raw = _important_terms(jd_text, limit=90)
    actionable_jd_terms = [
        term
        for term in jd_terms_raw
        if _is_actionable_keyword(term, jd_lower) or term in TOOL_TERMS or term in DOMAIN_TERMS or term in ROLE_SIGNAL_TERMS
    ]
    if not actionable_jd_terms:
        actionable_jd_terms = [term for term in jd_terms_raw if term not in STOPWORDS][:30]

    resume_terms = set(_important_terms(resume_text, limit=120))
    missing_terms = [term for term in actionable_jd_terms if term not in resume_terms][:18]
    matched_terms = [term for term in actionable_jd_terms if term in resume_terms][:18]
    overlap_ratio = (len(matched_terms) / max(len(actionable_jd_terms), 1)) if actionable_jd_terms else 0.0
    job_match = int(round(min(max(overlap_ratio, 0.0), 1.0) * 100))

    risks: list[RiskItem] = []
    fix_plan: list[FixPlanItem] = []
    hard_filter_hits: list[str] = []

    if ("visa" in jd_lower or "work authorization" in jd_lower or "citizenship" in jd_lower) and (
        "visa" not in resume_lower and "authorized" not in resume_lower and "citizen" not in resume_lower and "work authorization" not in resume_lower
    ):
        hard_filter_hits.append(_msg(locale, "hf_visa"))
    if ("bachelor" in jd_lower or "master" in jd_lower or "phd" in jd_lower or "degree" in jd_lower) and (
        "bachelor" not in resume_lower and "master" not in resume_lower and "phd" not in resume_lower and "degree" not in resume_lower
    ):
        hard_filter_hits.append(_msg(locale, "hf_degree"))
    if "security clearance" in jd_lower and "clearance" not in resume_lower:
        hard_filter_hits.append(_msg(locale, "hf_clearance"))

    years_required = max((int(x) for x in YEARS_RE.findall(jd_lower)), default=0)
    years_signal = _seniority_to_years(payload.candidate_profile.seniority)
    seniority_gap = years_required > 0 and years_required > years_signal + 2

    if seniority_gap:
        gap_years = max(1, years_required - years_signal)
        risks.append(RiskItem(type="seniority", severity="high" if years_required >= 8 else "medium", message=_msg(locale, "risk_seniority")))
        fix_plan.append(
            FixPlanItem(
                id="seniority-signal",
                title=_msg(locale, "fix_seniority_title"),
                impact_score=min(94, 48 + (gap_years * 8)),
                effort_minutes=min(55, 14 + (gap_years * 3)),
                reason=_msg(locale, "fix_seniority_reason"),
            )
        )

    if hard_filter_hits:
        risks.append(RiskItem(type="hard_filter", severity="high", message=_msg(locale, "risk_hard_filter", detail=", ".join(hard_filter_hits))))

    parsing_penalty = _parsing_penalty(
        resume_text,
        layout_profile=layout_profile,
        layout_fit=layout_fit,
    )

    if parsing_penalty >= 12:
        risks.append(RiskItem(type="parsing", severity="high" if parsing_penalty >= 20 else "medium", message=_msg(locale, "risk_parsing")))
        fix_plan.append(
            FixPlanItem(
                id="parsing-format",
                title=_msg(locale, "fix_parsing_title"),
                impact_score=min(92, 45 + (parsing_penalty * 2)),
                effort_minutes=min(45, 10 + (parsing_penalty // 2)),
                reason=_msg(locale, "fix_parsing_reason"),
            )
        )

    if len(missing_terms) >= 6:
        risks.append(RiskItem(type="keyword_gap", severity="high" if len(missing_terms) >= 12 else "medium", message=_msg(locale, "risk_keyword_gap")))
        fix_plan.append(
            FixPlanItem(
                id="keyword-priority",
                title=_msg(locale, "fix_keywords_title"),
                impact_score=min(95, 50 + (len(missing_terms) * 2)),
                effort_minutes=min(60, 8 + len(missing_terms)),
                reason=_msg(locale, "fix_keywords_reason"),
            )
        )

    numeric_evidence_count = len(re.findall(r"\d", resume_text))
    if numeric_evidence_count < 6:
        risks.append(RiskItem(type="evidence_gap", severity="medium", message=_msg(locale, "risk_evidence_gap")))
        missing_evidence_points = 6 - numeric_evidence_count
        fix_plan.append(
            FixPlanItem(
                id="evidence-bullets",
                title=_msg(locale, "fix_evidence_title"),
                impact_score=min(88, 45 + (missing_evidence_points * 6)),
                effort_minutes=min(50, 12 + (missing_evidence_points * 4)),
                reason=_msg(locale, "fix_evidence_reason"),
            )
        )

    ats_readability = max(20, min(100, 92 - parsing_penalty - (3 if len(missing_terms) > 10 else 0)))
    recommendation: Recommendation
    if any(r.type == "hard_filter" and r.severity == "high" for r in risks):
        recommendation = "skip"
    elif job_match >= 72 and ats_readability >= 70 and not any(r.severity == "high" for r in risks):
        recommendation = "apply"
    else:
        recommendation = "fix"

    high_risks = sum(1 for risk in risks if risk.severity == "high")
    confidence = max(0.4, min(0.95, 0.65 + (job_match / 250) - (high_risks * 0.08)))
    if recommendation == "skip":
        confidence = min(confidence, 0.55)
    if not fix_plan:
        fix_plan.append(FixPlanItem(id="final-polish", title=_msg(locale, "fix_evidence_title"), impact_score=25, effort_minutes=12, reason=_msg(locale, "fix_evidence_reason")))

    generation_mode = "heuristic"
    generation_scope = "heuristic"
    analysis_summary = ""

    llm_payload = json_completion(
        system_prompt=(
            "You are a senior recruiting analyst and ATS optimization specialist. "
            "Produce a realistic structured analysis from resume + JD. "
            "Avoid generic advice and use evidence from the provided text. "
            "Return strict JSON only."
        ),
        user_prompt=(
            f"Language: {_locale_language_name(locale)}.\n"
            "Analyze the candidate against the job and produce actionable outputs.\n"
            "Return JSON schema:\n"
            "{"
            "\"analysis_summary\":\"...\","
            "\"recommendation\":\"apply|fix|skip\","
            "\"confidence\":0.0,"
            "\"scores\":{\"job_match\":0,\"ats_readability\":0},"
            "\"risks\":[{\"type\":\"hard_filter|keyword_gap|parsing|seniority|evidence_gap\",\"severity\":\"low|medium|high\",\"message\":\"...\"}],"
            "\"fix_plan\":[{\"id\":\"...\",\"title\":\"...\",\"impact_score\":0,\"effort_minutes\":0,\"reason\":\"...\"}]"
            "}\n"
            "Rules:\n"
            "- confidence must be 0.40 to 0.95\n"
            "- job_match and ats_readability must be 0 to 100 integers\n"
            "- Return 2 to 5 risks and 2 to 5 fix_plan items\n"
            "- Make fixes concrete and evidence-driven, not generic\n"
            "- If hard filters are present, recommendation should usually be skip\n\n"
            f"Resume excerpt:\n{resume_text[:3000]}\n\n"
            f"JD excerpt:\n{jd_text[:3000]}\n\n"
            "Heuristic baseline (for reference, do not blindly copy):\n"
            f"recommendation={recommendation}, confidence={round(confidence, 2)}, "
            f"job_match={job_match}, ats_readability={ats_readability}\n"
            f"detected_hard_filters={hard_filter_hits}\n"
            f"matched_terms={matched_terms[:15]}\n"
            f"missing_terms={missing_terms[:15]}\n"
            f"heuristic_risks={[{'type': r.type, 'severity': r.severity, 'message': r.message} for r in risks]}\n"
            f"heuristic_fix_plan={[{'id': p.id, 'title': p.title, 'impact_score': p.impact_score, 'effort_minutes': p.effort_minutes, 'reason': p.reason} for p in fix_plan]}\n"
        ),
        temperature=0.22,
        max_output_tokens=1300,
    )

    if llm_payload:
        llm_summary = _safe_str(llm_payload.get("analysis_summary"), max_len=500)
        llm_risks = _safe_risk_items(llm_payload.get("risks"))
        llm_fixes = _safe_fix_plan_items(llm_payload.get("fix_plan"))
        llm_scores = llm_payload.get("scores") if isinstance(llm_payload.get("scores"), dict) else {}

        if llm_summary:
            analysis_summary = llm_summary
        if llm_risks:
            risks = llm_risks
        if llm_fixes:
            fix_plan = llm_fixes

        job_match = _clamp_int(
            llm_scores.get("job_match") if isinstance(llm_scores, dict) else None,
            default=job_match,
            min_value=0,
            max_value=100,
        )
        ats_readability = _clamp_int(
            llm_scores.get("ats_readability") if isinstance(llm_scores, dict) else None,
            default=ats_readability,
            min_value=0,
            max_value=100,
        )
        recommendation = _safe_recommendation(llm_payload.get("recommendation"), recommendation)
        confidence = _clamp_float(llm_payload.get("confidence"), default=confidence, min_value=0.4, max_value=0.95)

        generation_mode = "llm"
        generation_scope = "full-analysis"

    if hard_filter_hits and not any(r.type == "hard_filter" for r in risks):
        risks.insert(0, RiskItem(type="hard_filter", severity="high", message=_msg(locale, "risk_hard_filter", detail=", ".join(hard_filter_hits))))

    if recommendation == "apply" and any(r.type == "hard_filter" and r.severity == "high" for r in risks):
        recommendation = "skip"

    # If full analysis was unavailable, still try an LLM clarity rewrite for risk/fix text.
    if generation_scope == "heuristic":
        rewrite_payload = json_completion(
            system_prompt=(
                "You are a resume-job match explainer. "
                "Improve clarity and actionability only. "
                "Do not change numeric scores, risk types, severity, or recommendation. "
                "Return strict JSON."
            ),
            user_prompt=(
                f"Language: {_locale_language_name(locale)}.\n"
                "Rewrite messages for candidate clarity based on deterministic analysis.\n"
                "Return JSON schema:\n"
                "{"
                "\"analysis_summary\":\"...\","
                "\"risks\":[{\"type\":\"hard_filter|keyword_gap|parsing|seniority|evidence_gap\",\"message\":\"...\"}],"
                "\"fix_plan\":[{\"id\":\"...\",\"title\":\"...\",\"reason\":\"...\"}]"
                "}\n"
                "Keep risk count and fix count unchanged. Keep each message concise and concrete.\n\n"
                f"Resume excerpt:\n{resume_text[:3000]}\n\n"
                f"JD excerpt:\n{jd_text[:3000]}\n\n"
                f"Recommendation: {recommendation}\n"
                f"Scores: job_match={job_match}, ats_readability={ats_readability}\n"
                f"Risks: {[{'type': r.type, 'severity': r.severity, 'message': r.message} for r in risks]}\n"
                f"Fix plan: {[{'id': p.id, 'title': p.title, 'reason': p.reason} for p in fix_plan]}\n"
            ),
            temperature=0.15,
            max_output_tokens=850,
        )
        if rewrite_payload:
            rewrite_summary = _safe_str(rewrite_payload.get("analysis_summary"), max_len=400)
            rewrite_risks = rewrite_payload.get("risks")
            rewrite_fixes = rewrite_payload.get("fix_plan")

            if isinstance(rewrite_risks, list):
                risk_by_type: dict[str, str] = {}
                for item in rewrite_risks:
                    if not isinstance(item, dict):
                        continue
                    r_type = _safe_str(item.get("type"), max_len=40)
                    r_msg = _safe_str(item.get("message"), max_len=240)
                    if r_type and r_msg:
                        risk_by_type[r_type] = r_msg
                if risk_by_type:
                    risks = [
                        RiskItem(type=risk.type, severity=risk.severity, message=risk_by_type.get(risk.type, risk.message))
                        for risk in risks
                    ]

            if isinstance(rewrite_fixes, list):
                fix_by_id: dict[str, dict[str, str]] = {}
                for item in rewrite_fixes:
                    if not isinstance(item, dict):
                        continue
                    fix_id = _safe_str(item.get("id"), max_len=80)
                    title = _safe_str(item.get("title"), max_len=120)
                    reason = _safe_str(item.get("reason"), max_len=220)
                    if fix_id:
                        fix_by_id[fix_id] = {"title": title, "reason": reason}
                if fix_by_id:
                    updated_fix_plan: list[FixPlanItem] = []
                    for fix in fix_plan:
                        override = fix_by_id.get(fix.id, {})
                        updated_fix_plan.append(
                            FixPlanItem(
                                id=fix.id,
                                title=override.get("title") or fix.title,
                                impact_score=fix.impact_score,
                                effort_minutes=fix.effort_minutes,
                                reason=override.get("reason") or fix.reason,
                            )
                        )
                    fix_plan = updated_fix_plan

            if rewrite_summary:
                analysis_summary = rewrite_summary

            generation_mode = "llm"
            generation_scope = "rewrite-only"

    matched_term_evidence, missing_term_context, hard_filter_evidence = _term_evidence_maps(
        resume_text=resume_text,
        jd_text=jd_text,
        matched_terms=matched_terms,
        missing_terms=missing_terms,
        hard_filter_hits=hard_filter_hits,
    )
    if not analysis_summary:
        lead_risk = risks[0].message if risks else _msg(locale, f"recommend_{recommendation}")
        analysis_summary = _safe_str(
            f"Decision: {_msg(locale, f'recommend_{recommendation}')}. Primary evidence: {lead_risk}",
            max_len=320,
        )
    quality_samples = [analysis_summary] + [risk.message for risk in risks] + [item.reason for item in fix_plan]
    _ensure_quality_generation(
        tool_slug="base-analysis",
        generation_mode=generation_mode,
        generation_scope=generation_scope,
        sample_texts=quality_samples,
    )

    _emit_progress(
        progress_callback,
        stage="extracting_skills",
        label="Extracting your skills",
        percent=72,
        detail="Building hard/soft skill alignment from resume and JD.",
    )
    skills_comparison = _build_skills_comparison(resume_text, jd_text, matched_terms, missing_terms)
    searchability = _build_searchability(resume_text)
    recruiter_tips = _build_recruiter_tips(
        resume_text, jd_text, years_required, payload.candidate_profile.seniority,
    )

    return {
        "scores": ScoreCard(job_match=job_match, ats_readability=ats_readability),
        "recommendation": recommendation,
        "confidence": round(confidence, 2),
        "risks": risks,
        "fix_plan": sorted(fix_plan, key=lambda x: (x.impact_score, -x.effort_minutes), reverse=True),
        "missing_terms": missing_terms,
        "matched_terms": matched_terms,
        "hard_filter_hits": hard_filter_hits,
        "generation_mode": generation_mode,
        "generation_scope": generation_scope,
        "analysis_summary": analysis_summary,
        "skills_comparison": skills_comparison,
        "searchability": searchability,
        "recruiter_tips": recruiter_tips,
        "layout_profile": layout_profile,
        "layout_fit_for_target": layout_fit,
        "format_recommendation": layout_fit.get("format_recommendation", ""),
        "resume_file_meta": resume_file_meta,
        "matched_term_evidence": matched_term_evidence,
        "missing_term_context": missing_term_context,
        "hard_filter_evidence": hard_filter_evidence,
    }


def run_job_match(payload: ToolRequest) -> ToolResponse:
    base = _build_base_analysis(payload)
    return ToolResponse(
        recommendation=base["recommendation"],
        confidence=base["confidence"],
        scores=base["scores"],
        risks=base["risks"],
        fix_plan=base["fix_plan"],
        generated_at=datetime.now(timezone.utc),
        details={
            "hard_filters": base["hard_filter_hits"],
            "soft_match": {"matched_keywords": base["matched_terms"], "missing_keywords": base["missing_terms"]},
            "matched_term_evidence": base["matched_term_evidence"],
            "missing_term_context": base["missing_term_context"],
            "hard_filter_evidence": base["hard_filter_evidence"],
            "recommendation_label": _msg(payload.locale, f"recommend_{base['recommendation']}"),
            "layout_analysis": base["layout_profile"],
            "layout_fit_for_target": base["layout_fit_for_target"],
            "format_recommendation": base["format_recommendation"],
            "generation_mode": base["generation_mode"],
            "generation_scope": base["generation_scope"],
            "analysis_summary": base["analysis_summary"],
            "skills_comparison": base["skills_comparison"],
            "searchability": base["searchability"],
            "recruiter_tips": base["recruiter_tips"],
        },
    )


def _group_keyword(term: str) -> str:
    if term in HARD_FILTER_TERMS:
        return "hard_filters"
    if term in TOOL_TERMS:
        return "tooling"
    if term in DOMAIN_TERMS:
        return "domain"
    return "core_role"

def run_missing_keywords(payload: ToolRequest) -> ToolResponse:
    base = _build_base_analysis(payload)
    locale = payload.locale
    jd_lower = payload.job_description_text.lower()
    grouped: dict[str, list[str]] = {"hard_filters": [], "core_role": [], "tooling": [], "domain": []}
    for term in base["missing_terms"]:
        grouped[_group_keyword(term)].append(term)

    actionable_terms = [
        term
        for term in base["missing_terms"]
        if (
            term not in HARD_FILTER_TERMS
            and term not in WORK_MODE_TERMS
            and term not in LOW_SIGNAL_KEYWORD_TERMS
            and _is_actionable_keyword(term, jd_lower)
        )
    ]

    suggestions: list[dict[str, str]] = []
    for term in actionable_terms[:10]:
        group = _group_keyword(term)
        jd_evidence = (base.get("missing_term_context") or {}).get(term, [])
        evidence_hint = jd_evidence[0] if jd_evidence else ""
        if group == "tooling":
            section = "skills + experience"
            guidance = f"{_msg(locale, 'insert_skill', term=term)} {_msg(locale, 'insert_exp', term=term)}"
        elif group == "domain":
            section = "experience"
            guidance = _msg(locale, "insert_exp", term=term)
        else:
            section = "experience"
            guidance = _msg(locale, "insert_exp", term=term)
        if evidence_hint:
            guidance = f"{guidance} JD evidence: {evidence_hint}"
        suggestions.append({"keyword": term, "insert_in": section, "guidance": guidance})

    used_llm_suggestions = False
    llm_payload = json_completion(
        system_prompt=(
            "You are a resume keyword optimization assistant. "
            "Create practical insertion guidance and avoid generic or non-actionable terms. "
            "Return strict JSON only."
        ),
        user_prompt=(
            f"Language: {_locale_language_name(payload.locale)}.\n"
            "Given resume and job description, generate high-quality insertion suggestions.\n"
            "Use ONLY these candidate missing terms:\n"
            f"{actionable_terms[:12]}\n\n"
            "Return JSON schema:\n"
            "{"
            "\"insertion_suggestions\":["
            "{\"keyword\":\"...\",\"insert_in\":\"skills|experience|summary\",\"guidance\":\"...\",\"action\":\"add|skip\"}"
            "]"
            "}\n"
            "Rules:\n"
            "- Suggest 3 to 8 items\n"
            "- Exclude weak terms like schedule/location words\n"
            "- guidance must be concrete and evidence-based\n"
            "- If a term should not be inserted, mark action=skip\n\n"
            f"Resume excerpt:\n{payload.resume_text[:3000]}\n\n"
            f"JD excerpt:\n{payload.job_description_text[:3000]}\n"
        ),
        temperature=0.15,
        max_output_tokens=900,
    )
    if llm_payload and isinstance(llm_payload.get("insertion_suggestions"), list):
        allowed_terms = set(actionable_terms)
        llm_suggestions: list[dict[str, str]] = []
        seen_keywords: set[str] = set()
        for item in llm_payload["insertion_suggestions"]:
            if not isinstance(item, dict):
                continue
            keyword = _safe_str(item.get("keyword"), max_len=80).lower()
            if keyword not in allowed_terms:
                continue
            if keyword in seen_keywords:
                continue
            action = _safe_str(item.get("action"), max_len=12).lower() or "add"
            if action == "skip":
                continue
            insert_in = _safe_str(item.get("insert_in"), max_len=40).lower()
            if insert_in not in {"skills", "experience", "summary"}:
                insert_in = "experience"
            guidance = _safe_str(item.get("guidance"), max_len=280)
            if not guidance:
                continue
            llm_suggestions.append({"keyword": keyword, "insert_in": insert_in, "guidance": guidance})
            seen_keywords.add(keyword)
            if len(llm_suggestions) >= 8:
                break
        if llm_suggestions:
            suggestions = llm_suggestions
            used_llm_suggestions = True

    if _strict_llm_required():
        if not used_llm_suggestions or len(suggestions) < 3:
            raise QualityEnforcementError(
                "AI quality mode requires evidence-backed keyword insertion guidance. Please retry with a fuller job description.",
                status_code=503,
            )
        _ensure_quality_generation(
            tool_slug="missing-keywords",
            generation_mode="llm",
            generation_scope="full-analysis",
            sample_texts=[item.get("guidance", "") for item in suggestions],
        )

    stuffing = _keyword_stuffing_report(payload.resume_text, base["matched_terms"] + base["missing_terms"])

    return ToolResponse(
        recommendation=base["recommendation"],
        confidence=base["confidence"],
        scores=base["scores"],
        risks=base["risks"],
        fix_plan=base["fix_plan"],
        generated_at=datetime.now(timezone.utc),
        details={
            "keyword_groups": grouped,
            "insertion_suggestions": suggestions,
            "excluded_non_actionable_terms": [term for term in base["missing_terms"] if term not in actionable_terms][:12],
            "matched_term_evidence": base["matched_term_evidence"],
            "missing_term_context": base["missing_term_context"],
            "hard_filter_evidence": base["hard_filter_evidence"],
            "keyword_stuffing_detector": stuffing,
            "layout_analysis": base["layout_profile"],
            "layout_fit_for_target": base["layout_fit_for_target"],
            "format_recommendation": base["format_recommendation"],
            "generation_mode": base["generation_mode"],
            "generation_scope": base["generation_scope"],
            "analysis_summary": base["analysis_summary"],
            "skills_comparison": base["skills_comparison"],
            "searchability": base["searchability"],
            "recruiter_tips": base["recruiter_tips"],
        },
    )


def run_ats_checker(
    payload: ToolRequest,
    *,
    progress_callback: ProgressCallback | None = None,
) -> ToolResponse:
    _emit_progress(
        progress_callback,
        stage="parsing_resume",
        label="Parsing your resume",
        percent=16,
        detail="Reading resume structure, layout profile, and extractable fields.",
    )
    base = _build_base_analysis(payload, progress_callback=progress_callback)
    resume = payload.resume_text
    lines = [line.strip() for line in resume.splitlines() if line.strip()]
    credibility = _credibility_score(payload.resume_text, payload.job_description_text)
    layout_profile = base["layout_profile"]
    layout_fit = base["layout_fit_for_target"]
    effective_layout = _effective_detected_layout(layout_profile, resume)
    strong_layout = _has_strong_layout_evidence(layout_profile)
    layout_confidence = _clamp_float(layout_profile.get("confidence"), default=0.0, min_value=0.0, max_value=1.0)
    text_signals = _text_layout_signals(resume)
    table_count = _clamp_int(layout_profile.get("table_count"), default=0, min_value=0, max_value=200)

    parsing_flags: list[str] = []
    if text_signals["probable_table"] or table_count > 0:
        parsing_flags.append(_msg(payload.locale, "flag_table"))
    if (
        (
            effective_layout == "multi_column"
            and (strong_layout or layout_confidence >= 0.72)
        )
        or (
            effective_layout == "hybrid"
            and (strong_layout or layout_confidence >= 0.62)
        )
        or (text_signals["wide_space_lines"] >= 12 and text_signals["tab_line_count"] >= 4)
    ):
        parsing_flags.append(_msg(payload.locale, "flag_multicol"))
    if (
        "header" in resume.lower()
        or "footer" in resume.lower()
        or _clamp_float(layout_profile.get("header_link_density"), default=0.0, min_value=0.0, max_value=1.0) >= 0.5
    ):
        parsing_flags.append(_msg(payload.locale, "flag_header"))
    _emit_progress(
        progress_callback,
        stage="generating_recommendations",
        label="Generating recommendations",
        percent=90,
        detail="Building ATS report checks, issue evidence, and fix guidance.",
    )
    ats_report = _build_ats_report(
        payload=payload,
        base=base,
        parsing_flags=parsing_flags,
        credibility=credibility,
        lines=lines,
    )

    details = {
        "extraction_preview": {
            "name": lines[0] if lines else "",
            "email": EMAIL_RE.search(resume).group(0) if EMAIL_RE.search(resume) else "",
            "phone": PHONE_RE.search(resume).group(0) if PHONE_RE.search(resume) else "",
            "top_skills": _important_terms(resume, limit=12),
        },
        "ats_report": ats_report,
        "parsing_flags": parsing_flags,
        "matched_term_evidence": base["matched_term_evidence"],
        "missing_term_context": base["missing_term_context"],
        "hard_filter_evidence": base["hard_filter_evidence"],
        "layout_analysis": layout_profile,
        "layout_fit_for_target": layout_fit,
        "format_recommendation": base["format_recommendation"],
        "resume_file_meta": base["resume_file_meta"],
        "resume_credibility_score": credibility,
        "generation_mode": base["generation_mode"],
        "generation_scope": base["generation_scope"],
        "analysis_summary": base["analysis_summary"],
        "skills_comparison": base["skills_comparison"],
        "searchability": base["searchability"],
        "recruiter_tips": base["recruiter_tips"],
    }
    response = ToolResponse(
        recommendation=base["recommendation"],
        confidence=base["confidence"],
        scores=base["scores"],
        risks=base["risks"],
        fix_plan=base["fix_plan"],
        generated_at=datetime.now(timezone.utc),
        details=details,
    )
    _emit_progress(
        progress_callback,
        stage="completed",
        label="Analysis complete",
        percent=100,
        detail="ATS report is ready.",
    )
    return response


def run_cover_letter(payload: ToolRequest) -> ToolResponse:
    base = _build_base_analysis(payload)
    locale = payload.locale
    mode_map = {
        "recruiter": _msg(locale, "mode_recruiter"),
        "hr": _msg(locale, "mode_hr"),
        "technical": _msg(locale, "mode_technical"),
    }
    top_match = ", ".join(base["matched_terms"][:3]) or "relevant experience"
    role_hint = ", ".join(base["matched_terms"][:2]) or "business impact"
    improvement = ", ".join(base["missing_terms"][:2]) or "role terms"

    letters = {}
    for mode_key, mode_label in mode_map.items():
        letters[mode_key] = (
            f"{mode_label}\n\n{_msg(locale, 'cover_greeting')}\n"
            f"{_msg(locale, 'cover_p1', top_match=top_match)}\n\n"
            f"{_msg(locale, 'cover_p2', role_hint=role_hint, improvement=improvement)}\n\n"
            f"{_msg(locale, 'cover_closing')}"
        )
    generation_mode = "heuristic"
    generation_scope = "heuristic"

    llm_payload = json_completion(
        system_prompt=(
            "You write concise job application cover letters. "
            "Return strict JSON only."
        ),
        user_prompt=(
            f"Language: {_locale_language_name(locale)}.\n"
            "Create 3 versions of a cover letter using only user-provided evidence.\n"
            "Do not invent companies, years, certifications, or outcomes.\n"
            "Each version must be 80-140 words and practical.\n"
            "JSON schema:\n"
            "{"
            "\"letters\": {\"recruiter\": \"...\", \"hr\": \"...\", \"technical\": \"...\"},"
            "\"default_mode\": \"technical\""
            "}\n\n"
            f"Resume:\n{payload.resume_text[:3500]}\n\n"
            f"Job description:\n{payload.job_description_text[:3500]}\n\n"
            f"Matched terms: {', '.join(base['matched_terms'][:12])}\n"
            f"Missing terms: {', '.join(base['missing_terms'][:12])}\n"
            f"Recommendation: {base['recommendation']}\n"
        ),
        temperature=0.25,
        max_output_tokens=1200,
    )
    if llm_payload:
        raw_letters = llm_payload.get("letters")
        if isinstance(raw_letters, dict):
            recruiter = _safe_str(raw_letters.get("recruiter"), max_len=2200)
            hr = _safe_str(raw_letters.get("hr"), max_len=2200)
            technical = _safe_str(raw_letters.get("technical"), max_len=2200)
            if recruiter and hr and technical:
                letters = {
                    "recruiter": recruiter,
                    "hr": hr,
                    "technical": technical,
                }
                generation_mode = "llm"
                generation_scope = "full-analysis"

    _ensure_quality_generation(
        tool_slug="cover-letter",
        generation_mode=generation_mode,
        generation_scope=generation_scope,
        sample_texts=[letters.get("recruiter", ""), letters.get("hr", ""), letters.get("technical", "")],
    )

    humanization = {
        mode: _humanization_report(text)
        for mode, text in letters.items()
    }

    return ToolResponse(
        recommendation=base["recommendation"],
        confidence=base["confidence"],
        scores=base["scores"],
        risks=base["risks"],
        fix_plan=base["fix_plan"],
        generated_at=datetime.now(timezone.utc),
        details={
            "letters": letters,
            "default_mode": "technical",
            "mode_labels": mode_map,
            "humanization_filter": humanization,
            "layout_analysis": base["layout_profile"],
            "layout_fit_for_target": base["layout_fit_for_target"],
            "format_recommendation": base["format_recommendation"],
            "generation_mode": generation_mode,
            "generation_scope": generation_scope,
            "skills_comparison": base["skills_comparison"],
            "searchability": base["searchability"],
            "recruiter_tips": base["recruiter_tips"],
        },
    )


def run_interview_predictor(payload: ToolRequest) -> ToolResponse:
    base = _build_base_analysis(payload)
    locale = payload.locale
    questions: list[dict[str, str]] = []

    for term in base["missing_terms"][:4]:
        questions.append({
            "question": _msg(locale, "interview_missing_q", term=term),
            "reason": _msg(locale, "interview_missing_r"),
            "framework": _msg(locale, "framework_star"),
        })
    if any(r.type == "seniority" for r in base["risks"]):
        questions.append({
            "question": _msg(locale, "interview_seniority_q"),
            "reason": _msg(locale, "interview_seniority_r"),
            "framework": _msg(locale, "framework_star_tradeoff"),
        })
    if not questions:
        questions.append({
            "question": _msg(locale, "interview_fallback_q"),
            "reason": _msg(locale, "interview_fallback_r"),
            "framework": _msg(locale, "framework_star"),
        })
    red_flags = [_msg(locale, "red_flag_1"), _msg(locale, "red_flag_2")]
    generation_mode = "heuristic"
    generation_scope = "heuristic"

    llm_payload = json_completion(
        system_prompt=(
            "You are an interview preparation assistant. "
            "Return strict JSON only."
        ),
        user_prompt=(
            f"Language: {_locale_language_name(locale)}.\n"
            "Generate interview prep from resume vs job description.\n"
            "Return only JSON schema:\n"
            "{"
            "\"predicted_questions\": [{\"question\":\"...\",\"reason\":\"...\",\"framework\":\"STAR|STAR + tradeoff|Technical deep dive\"}],"
            "\"red_flag_preview\": [\"...\", \"...\"]"
            "}\n"
            "Provide 4-6 predicted questions. Keep each question under 25 words.\n"
            "Do not invent facts. Use only user-provided information.\n\n"
            f"Resume:\n{payload.resume_text[:3500]}\n\n"
            f"Job description:\n{payload.job_description_text[:3500]}\n\n"
            f"Matched terms: {', '.join(base['matched_terms'][:12])}\n"
            f"Missing terms: {', '.join(base['missing_terms'][:12])}\n"
            f"Risks: {', '.join([f'{r.type}:{r.severity}' for r in base['risks']])}\n"
        ),
        temperature=0.2,
        max_output_tokens=1000,
    )
    if llm_payload:
        parsed_questions = _safe_question_items(llm_payload.get("predicted_questions"), max_items=6)
        parsed_red_flags = _safe_str_list(llm_payload.get("red_flag_preview"), max_items=4, max_len=200)
        if parsed_questions:
            questions = parsed_questions
            if parsed_red_flags:
                red_flags = parsed_red_flags
            generation_mode = "llm"
            generation_scope = "full-analysis"

    _ensure_quality_generation(
        tool_slug="interview-predictor",
        generation_mode=generation_mode,
        generation_scope=generation_scope,
        sample_texts=[item.get("question", "") for item in questions] + [item.get("reason", "") for item in questions],
    )

    return ToolResponse(
        recommendation=base["recommendation"],
        confidence=base["confidence"],
        scores=base["scores"],
        risks=base["risks"],
        fix_plan=base["fix_plan"],
        generated_at=datetime.now(timezone.utc),
        details={
            "predicted_questions": questions,
            "red_flag_preview": red_flags,
            "layout_analysis": base["layout_profile"],
            "layout_fit_for_target": base["layout_fit_for_target"],
            "format_recommendation": base["format_recommendation"],
            "generation_mode": generation_mode,
            "generation_scope": generation_scope,
            "skills_comparison": base["skills_comparison"],
            "searchability": base["searchability"],
            "recruiter_tips": base["recruiter_tips"],
        },
    )


def save_lead(payload: LeadCaptureRequest) -> str:
    logger.info(
        "tools_lead_capture session=%s email=%s tool=%s consent=%s locale=%s",
        payload.session_id,
        payload.email,
        payload.tool,
        payload.consent,
        payload.locale,
    )
    return _msg(payload.locale, "lead_saved")

ADDITIONAL_TOOL_SLUGS = {
    "one-click-optimize",
    "resume-score",
    "resume-summary-generator",
    "resume-bullet-points-generator",
    "ai-resume-tool",
    "job-application-tracker",
    "jobs",
    "linkedin-optimization-tool",
    "resume-builder-tool",
    "resume-optimization-report",
    "career-change-tool",
    "product-walkthrough",
    "job-application-roi-calculator",
    "seniority-calibration-tool",
    "rejection-reason-classifier",
    "cv-region-translator",
}

MERGED_INTO_RESUME_OPTIMIZATION_REPORT = {
    "one-click-optimize",
    "resume-score",
    "resume-summary-generator",
    "resume-bullet-points-generator",
    "ai-resume-tool",
    "linkedin-optimization-tool",
    "resume-builder-tool",
}

SUMMARIZER_TOOL_SLUGS = {
    "text-summarizer": "text",
    "word-summarizer": "word",
    "pdf-summarizer": "pdf",
    "ppt-summarizer": "ppt",
    "youtube-summarizer": "youtube",
    "video-summarizer": "video",
    "image-summarizer": "image",
}


def _normalize_public_url(raw_url: str, *, field_label: str) -> tuple[str, str]:
    value = (raw_url or "").strip()
    if not value:
        raise ValueError(f"{field_label} is required.")
    if not re.match(r"^https?://", value, flags=re.IGNORECASE):
        value = f"https://{value}"
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Only http/https {field_label.lower()}s are supported.")
    if not parsed.netloc:
        raise ValueError(f"Invalid {field_label.lower()}.")
    hostname = (parsed.hostname or "").lower().strip()
    if not hostname:
        raise ValueError(f"Invalid {field_label.lower()} host.")
    normalized = urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path or "/",
            "",
            parsed.query,
            "",
        )
    )
    return normalized, hostname


def _normalize_job_url(raw_url: str) -> tuple[str, str]:
    return _normalize_public_url(raw_url, field_label="Job URL")


def _normalize_resume_url(raw_url: str) -> tuple[str, str]:
    return _normalize_public_url(raw_url, field_label="Resume URL")


def _host_is_private_or_local(hostname: str) -> bool:
    host = (hostname or "").strip().lower()
    if host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return bool(ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved)
    except ValueError:
        pass
    try:
        for family, _socktype, _proto, _canon, sockaddr in socket.getaddrinfo(host, None):
            address = sockaddr[0] if sockaddr else ""
            if not address:
                continue
            try:
                resolved = ipaddress.ip_address(address)
            except ValueError:
                continue
            if resolved.is_private or resolved.is_loopback or resolved.is_link_local or resolved.is_reserved:
                return True
    except Exception:
        # If DNS resolution fails, we do not hard-block.
        return False
    return False


def _normalize_google_drive_url(parsed: Any) -> str | None:
    host = (parsed.netloc or "").lower()
    if "drive.google.com" not in host:
        return None
    path = parsed.path or ""
    query = parse_qs(parsed.query or "")
    file_id = ""
    match = re.search(r"/file/d/([^/]+)", path)
    if match:
        file_id = match.group(1)
    if not file_id:
        file_id = _safe_str((query.get("id") or [""])[0], max_len=200)
    if not file_id:
        return None
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _normalize_dropbox_url(parsed: Any) -> str | None:
    host = (parsed.netloc or "").lower()
    if "dropbox.com" not in host and "dropboxusercontent.com" not in host:
        return None
    path = parsed.path or ""
    if not path:
        return None
    query = parse_qs(parsed.query or "")
    query["dl"] = ["1"]
    rebuilt_query = "&".join(f"{key}={value}" for key, values in query.items() for value in values)
    netloc = parsed.netloc.lower().replace("www.", "")
    if "dropbox.com" in netloc:
        netloc = netloc.replace("dropbox.com", "dl.dropboxusercontent.com")
    return urlunparse(("https", netloc, path, "", rebuilt_query, ""))


def _normalize_onedrive_url(parsed: Any) -> str | None:
    host = (parsed.netloc or "").lower()
    if "1drv.ms" not in host and "onedrive.live.com" not in host:
        return None
    query = parse_qs(parsed.query or "")
    query["download"] = ["1"]
    rebuilt_query = "&".join(f"{key}={value}" for key, values in query.items() for value in values)
    return urlunparse(("https", parsed.netloc.lower(), parsed.path or "/", "", rebuilt_query, ""))


def _normalize_resume_download_url(url: str) -> str:
    parsed = urlparse(url)
    for resolver in (_normalize_google_drive_url, _normalize_dropbox_url, _normalize_onedrive_url):
        resolved = resolver(parsed)
        if resolved:
            return resolved
    return url


def _extract_filename_from_content_disposition(value: str) -> str:
    if not value:
        return ""
    # RFC 6266 fallback parsing.
    filename_star = re.search(r"filename\*=UTF-8''([^;]+)", value, flags=re.IGNORECASE)
    if filename_star:
        try:
            return _safe_str(re.sub(r"%([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), filename_star.group(1)), max_len=255)
        except Exception:
            pass
    filename = re.search(r'filename="?([^";]+)"?', value, flags=re.IGNORECASE)
    if filename:
        return _safe_str(filename.group(1).strip(), max_len=255)
    return ""


def _extension_from_content_type(content_type: str) -> str:
    sanitized = _safe_str(content_type.split(";")[0], max_len=120).lower()
    if not sanitized:
        return ""
    explicit = RESUME_CONTENT_TYPE_EXTENSION_HINTS.get(sanitized)
    if explicit:
        return explicit
    guessed = mimetypes.guess_extension(sanitized) or ""
    return guessed.lstrip(".").lower()


def _filename_from_url_and_headers(final_url: str, headers: Any) -> str:
    disposition = ""
    try:
        disposition = _safe_str(headers.get("content-disposition"), max_len=500)
    except Exception:
        disposition = ""
    filename = _extract_filename_from_content_disposition(disposition)
    if filename:
        return filename
    path_name = os.path.basename(urlparse(final_url).path or "").strip()
    return _safe_str(path_name, max_len=255)


def _extension_from_filename(filename: str) -> str:
    if "." not in filename:
        return ""
    return _safe_str(filename.rsplit(".", 1)[-1], max_len=20).lower()


def _safe_resume_filename(final_url: str, headers: Any, content_type: str) -> str:
    filename = _filename_from_url_and_headers(final_url, headers)
    ext = _extension_from_filename(filename)
    if not ext:
        guessed_ext = _extension_from_content_type(content_type)
        if guessed_ext:
            base = _safe_str(filename or "resume", max_len=220).rstrip(".")
            filename = f"{base}.{guessed_ext}"
            ext = guessed_ext
    if not filename:
        filename = "resume.txt"
    return filename


def _strip_html_fragment(value: str) -> str:
    if not value:
        return ""
    cleaned = HTML_TAG_RE.sub("\n", value)
    cleaned = html.unescape(cleaned)
    return _normalize_extracted_text(cleaned)


def _safe_json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return None


def _iter_json_nodes(payload: Any) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        nodes.append(payload)
        graph = payload.get("@graph")
        if isinstance(graph, list):
            nodes.extend(item for item in graph if isinstance(item, dict))
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                nodes.extend(_iter_json_nodes(item))
    return nodes


def _extract_location_text(value: Any) -> str:
    if isinstance(value, list):
        parts = [_extract_location_text(item) for item in value]
        return _safe_str(", ".join([part for part in parts if part]), max_len=180)
    if isinstance(value, dict):
        address = value.get("address")
        if isinstance(address, dict):
            parts = [
                _safe_str(address.get("addressLocality"), max_len=80),
                _safe_str(address.get("addressRegion"), max_len=80),
                _safe_str(address.get("addressCountry"), max_len=80),
            ]
            merged = ", ".join([part for part in parts if part])
            if merged:
                return _safe_str(merged, max_len=180)
        direct = _safe_str(value.get("name"), max_len=180)
        if direct:
            return direct
    if isinstance(value, str):
        return _safe_str(value, max_len=180)
    return ""


def _extract_jobposting_json_ld(soup: Any) -> dict[str, str]:
    if soup is None:
        return {}
    scripts = soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.IGNORECASE)})
    for script in scripts:
        raw_json = script.string or script.get_text() or ""
        parsed = _safe_json_loads(raw_json.strip())
        if parsed is None:
            continue
        for node in _iter_json_nodes(parsed):
            type_value = node.get("@type")
            types: list[str] = []
            if isinstance(type_value, str):
                types = [type_value.lower()]
            elif isinstance(type_value, list):
                types = [str(item).lower() for item in type_value]
            if "jobposting" not in {item.replace(" ", "") for item in types}:
                continue
            title = _safe_str(node.get("title"), max_len=180)
            company = ""
            hiring = node.get("hiringOrganization")
            if isinstance(hiring, dict):
                company = _safe_str(hiring.get("name"), max_len=180)
            location = _extract_location_text(node.get("jobLocation") or node.get("jobLocationType"))
            description = node.get("description")
            description_text = _strip_html_fragment(description if isinstance(description, str) else "")
            if description_text:
                return {
                    "title": title,
                    "company": company,
                    "location": location,
                    "description": description_text,
                }
    return {}


def _domain_specific_job_extract(domain: str, soup: Any) -> dict[str, str]:
    if soup is None:
        return {}
    selectors: list[str] = []
    parser_name = ""
    if "greenhouse.io" in domain:
        parser_name = "greenhouse"
        selectors = ["#content", ".content", "#app", "[data-qa='job-description']"]
    elif "lever.co" in domain:
        parser_name = "lever"
        selectors = [".posting-page", ".content", ".section-wrapper", "#content"]
    elif "myworkdayjobs.com" in domain or "workday" in domain:
        parser_name = "workday"
        selectors = ["[data-automation-id='jobPostingDescription']", "[data-automation-id='jobDetails']", "main"]
    elif "indeed.com" in domain:
        parser_name = "indeed"
        selectors = ["#jobDescriptionText", "[data-testid='jobsearch-JobComponent-description']", "main"]

    if not selectors:
        return {}

    text_blocks: list[str] = []
    for selector in selectors:
        try:
            nodes = soup.select(selector)
        except Exception:
            nodes = []
        if not nodes:
            continue
        for node in nodes[:5]:
            extracted = _normalize_extracted_text(node.get_text("\n", strip=True))
            if len(extracted) >= 120:
                text_blocks.append(extracted)
        if text_blocks:
            break

    if not text_blocks:
        return {}

    title = ""
    try:
        heading = soup.select_one("h1")
        title = _safe_str(heading.get_text(" ", strip=True) if heading else "", max_len=180)
    except Exception:
        title = ""

    return {
        "parser": parser_name,
        "title": title,
        "description": text_blocks[0],
    }


def _readability_job_extract(raw_html: str) -> dict[str, str]:
    text = ""
    title = ""
    try:
        from readability import Document  # type: ignore

        doc = Document(raw_html)
        title = _safe_str(doc.short_title(), max_len=180)
        summary_html = doc.summary(html_partial=True)
        text = _strip_html_fragment(summary_html)
    except Exception:
        text = ""

    if len(text) >= 120:
        return {"title": title, "description": text}

    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(raw_html, "html.parser")
        title = title or _safe_str(soup.title.get_text(" ", strip=True) if soup.title else "", max_len=180)
        text = _normalize_extracted_text(soup.get_text("\n", strip=True))
    except Exception:
        text = _normalize_extracted_text(_strip_html_fragment(raw_html))

    return {"title": title, "description": text}


def extract_job_from_url(payload: ExtractJobRequest) -> ExtractJobResponse:
    normalized_url, hostname = _normalize_job_url(payload.job_url)
    if _host_is_private_or_local(hostname):
        raise ValueError("Private or local URLs are not allowed for job extraction.")

    try:
        import httpx
    except Exception as exc:
        raise ValueError("Job extraction is unavailable because http client dependency is missing.") from exc

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    with httpx.Client(timeout=12.0, follow_redirects=True, headers=headers) as client:
        response = client.get(normalized_url)

    final_url = str(response.url)
    domain = (urlparse(final_url).hostname or hostname).lower()
    page_html = response.text or ""
    html_lower = page_html.lower()

    warnings: list[str] = []
    blocked = False
    if response.status_code in {401, 403, 429}:
        blocked = True
        warnings.append(f"Job page returned HTTP {response.status_code}.")
    if any(marker in html_lower for marker in JOB_AUTH_WALL_MARKERS):
        blocked = True
        warnings.append("Page appears protected (auth wall, captcha, or JS-only rendering).")

    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(page_html, "html.parser")
    except Exception:
        soup = None
        warnings.append("HTML parser dependency unavailable; using plain extraction fallback.")

    extraction_mode: str = "readability"
    title = ""
    company = ""
    location = ""
    description = ""

    json_ld = _extract_jobposting_json_ld(soup)
    if json_ld.get("description"):
        extraction_mode = "json_ld"
        title = _safe_str(json_ld.get("title"), max_len=180)
        company = _safe_str(json_ld.get("company"), max_len=180)
        location = _safe_str(json_ld.get("location"), max_len=180)
        description = _safe_str(json_ld.get("description"), max_len=60000)

    if len(description) < 220:
        domain_result = _domain_specific_job_extract(domain, soup)
        if domain_result.get("description"):
            extraction_mode = "domain_parser"
            title = title or _safe_str(domain_result.get("title"), max_len=180)
            description = _safe_str(domain_result.get("description"), max_len=60000)

    if len(description) < 220:
        readability_result = _readability_job_extract(page_html)
        if readability_result.get("description"):
            extraction_mode = "readability"
            title = title or _safe_str(readability_result.get("title"), max_len=180)
            description = _safe_str(readability_result.get("description"), max_len=60000)

    # Last pass cleanup and noise trimming.
    description = _normalize_extracted_text(description)
    if description:
        noisy_lines = {
            "cookie policy",
            "accept all cookies",
            "privacy notice",
            "terms of service",
            "all rights reserved",
        }
        cleaned_lines = [
            line
            for line in description.splitlines()
            if line.strip() and all(noise not in line.lower() for noise in noisy_lines)
        ]
        description = _normalize_extracted_text("\n".join(cleaned_lines))

    if len(description) < 180:
        blocked = True
        warnings.append("Page requires sign-in or dynamic rendering. Paste JD text manually.")

    warning_unique = list(dict.fromkeys(_safe_str(item, max_len=240) for item in warnings if item))
    return ExtractJobResponse(
        job_url=payload.job_url,
        normalized_url=final_url,
        domain=domain,
        title=title,
        company=company,
        location=location,
        job_description_text=description,
        characters=len(description),
        extraction_mode=extraction_mode if extraction_mode in {"json_ld", "domain_parser", "readability"} else "readability",
        warnings=warning_unique[:8],
        blocked=blocked,
    )


def extract_resume_from_url(payload: ExtractResumeUrlRequest) -> ExtractResumeUrlResponse:
    normalized_url, hostname = _normalize_resume_url(payload.resume_url)
    normalized_url = _normalize_resume_download_url(normalized_url)
    normalized_url, hostname = _normalize_resume_url(normalized_url)
    if _host_is_private_or_local(hostname):
        raise ValueError("Private or local URLs are not allowed for resume extraction.")

    try:
        import httpx
    except Exception as exc:
        raise ValueError("Resume URL extraction is unavailable because http client dependency is missing.") from exc

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,"
            "text/plain,text/rtf,text/markdown,image/*,*/*;q=0.8"
        ),
    }
    with httpx.Client(timeout=15.0, follow_redirects=True, headers=headers) as client:
        response = client.get(normalized_url)

    final_url = str(response.url)
    domain = (urlparse(final_url).hostname or hostname).lower()
    content_type = _safe_str(response.headers.get("content-type"), max_len=200).lower()
    body = response.content or b""
    text_html = (response.text or "").lower() if "text/html" in content_type else ""
    blocked = False
    warnings: list[str] = []
    if response.status_code in {401, 403, 404, 429}:
        blocked = True
        warnings.append(f"Resume URL returned HTTP {response.status_code}.")
    if text_html and any(marker in text_html for marker in RESUME_AUTH_WALL_MARKERS):
        blocked = True
        warnings.append("Resume link appears protected (auth wall, captcha, or blocked download).")
    if blocked:
        return ExtractResumeUrlResponse(
            resume_url=payload.resume_url,
            normalized_url=final_url,
            domain=domain,
            filename="",
            content_type=content_type or "application/octet-stream",
            resume_text="",
            characters=0,
            details={},
            blocked=True,
            warnings=list(dict.fromkeys(_safe_str(item, max_len=240) for item in warnings if item))[:8],
            content_base64=None,
        )

    if not body:
        raise ValueError("Resume URL returned an empty file.")
    if len(body) > 10 * 1024 * 1024:
        raise ValueError("Resume file from URL is too large. Maximum allowed size is 10 MB.")

    filename = _safe_resume_filename(final_url, response.headers, content_type)
    ext = _extension_from_filename(filename)
    if ext not in {"txt", "md", "rtf", "pdf", "docx", "doc", "png", "jpg", "jpeg", "webp", "gif", "bmp"}:
        raise ValueError(
            "Unsupported resume file type from URL. Use a direct link to .pdf, .docx, .txt, .rtf, .md, or image."
        )

    extracted = extract_text_from_file(filename=filename, content=body)
    details = dict(extracted.details)
    details["fetched_from_url"] = True
    details["fetched_domain"] = domain
    details["normalized_url"] = final_url
    warnings_clean = list(dict.fromkeys(_safe_str(item, max_len=240) for item in warnings if item))[:8]
    return ExtractResumeUrlResponse(
        resume_url=payload.resume_url,
        normalized_url=final_url,
        domain=domain,
        filename=extracted.filename,
        content_type=content_type or "application/octet-stream",
        resume_text=extracted.text,
        characters=extracted.characters,
        details=details,
        blocked=False,
        warnings=warnings_clean,
        content_base64=base64.b64encode(body).decode("ascii"),
    )


def _summarizer_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p.strip() for p in parts if p.strip()]


def _keyword_scores(sentences: list[str]) -> dict[str, int]:
    terms: list[str] = []
    for sentence in sentences:
        for term in _tokenize(sentence):
            if term not in STOPWORDS and len(term) > 2:
                terms.append(term)
    counts = Counter(terms)
    return dict(counts)


def _best_sentences(sentences: list[str], limit: int) -> list[str]:
    if not sentences:
        return []
    score_map = _keyword_scores(sentences)

    def sentence_score(value: str) -> int:
        return sum(score_map.get(term, 0) for term in _tokenize(value))

    ranked = sorted(sentences, key=sentence_score, reverse=True)
    seen: set[str] = set()
    result: list[str] = []
    for sentence in ranked:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(sentence)
        if len(result) >= limit:
            break
    return result


def _length_to_limits(length: str) -> tuple[int, int, int]:
    if length == "short":
        return (2, 4, 3)
    if length == "long":
        return (8, 10, 6)
    return (4, 6, 4)


_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}


def _extract_youtube_id(url: str) -> str | None:
    try:
        parsed = urlparse(url)
    except Exception:
        return None

    host = parsed.netloc.lower().split(":")[0]  # strip port if present
    if host not in _YOUTUBE_HOSTS:
        return None

    if host in ("youtu.be", "www.youtu.be"):
        value = parsed.path.strip("/")
        return value or None

    # youtube.com / www.youtube.com / m.youtube.com
    if parsed.path == "/watch":
        query = parse_qs(parsed.query)
        video_ids = query.get("v") or []
        return video_ids[0] if video_ids else None
    if parsed.path.startswith("/shorts/"):
        return parsed.path.split("/shorts/")[-1] or None
    if parsed.path.startswith("/embed/"):
        return parsed.path.split("/embed/")[-1] or None
    return None


def _detect_pdf_layout_profile(
    *,
    reader: Any,
    extracted_text: str,
) -> dict[str, Any]:
    line_starts: list[float] = []
    page_count = len(reader.pages)
    for page_index, page in enumerate(reader.pages):
        width = 0.0
        height = 0.0
        try:
            width = float(page.mediabox.width or 0.0)
        except Exception:
            width = 0.0
        try:
            height = float(page.mediabox.height or 0.0)
        except Exception:
            height = 0.0
        if width <= 0:
            width = 595.0
        if height <= 0:
            height = 842.0

        row_min_x: dict[tuple[int, int], float] = {}
        row_token_count: dict[tuple[int, int], int] = {}

        def visitor(text: str, _cm: Any, tm: Any, _font_dict: Any, _font_size: Any) -> None:
            value = (text or "").strip()
            if not value or len(value) < 2:
                return
            try:
                x = float(tm[4]) if tm and len(tm) > 5 else 0.0
                y = float(tm[5]) if tm and len(tm) > 5 else 0.0
            except Exception:
                x = 0.0
                y = 0.0
            if x <= 0 or y <= 0:
                return
            x_norm = max(0.0, min(1.0, x / width))
            y_norm = max(0.0, min(1.0, y / height))
            row_bucket = int(round((1.0 - y_norm) * 120))
            key = (page_index, row_bucket)
            previous = row_min_x.get(key)
            row_min_x[key] = x_norm if previous is None else min(previous, x_norm)
            row_token_count[key] = row_token_count.get(key, 0) + 1

        try:
            page.extract_text(visitor_text=visitor)
        except Exception:
            continue

        for key, start_x in row_min_x.items():
            if row_token_count.get(key, 0) >= 2:
                line_starts.append(start_x)

    profile = _infer_layout_profile_from_text(extracted_text, source_type="pdf")
    total = len(line_starts)
    if total >= 24:
        left = sum(1 for value in line_starts if value <= 0.40)
        middle = sum(1 for value in line_starts if 0.40 < value < 0.57)
        right = sum(1 for value in line_starts if value >= 0.57)
        left_ratio = left / total
        right_ratio = right / total
        middle_ratio = middle / total
        if left_ratio >= 0.38 and right_ratio >= 0.18 and right >= 8 and middle_ratio <= 0.26:
            profile["detected_layout"] = "multi_column"
            profile["column_count"] = 2
            profile["confidence"] = max(float(profile.get("confidence", 0.6)), 0.82)
            profile["signals"] = list(dict.fromkeys([*profile.get("signals", []), "pdf_two_column_x_bands"]))
        elif left_ratio >= 0.34 and right_ratio >= 0.10 and right >= 5 and middle_ratio <= 0.34:
            profile["detected_layout"] = "hybrid"
            profile["column_count"] = max(2, int(profile.get("column_count", 1)))
            profile["confidence"] = max(float(profile.get("confidence", 0.55)), 0.68)
            profile["signals"] = list(dict.fromkeys([*profile.get("signals", []), "pdf_mixed_x_distribution"]))

    profile["signals"] = list(dict.fromkeys([*profile.get("signals", []), f"pdf_pages_{page_count}", f"pdf_line_starts_{total}"]))[:20]
    profile["source_type"] = "pdf"
    return profile


def _detect_docx_layout_profile(
    *,
    content: bytes,
    extracted_text: str,
) -> dict[str, Any]:
    profile = _infer_layout_profile_from_text(extracted_text, source_type="word")
    try:
        with ZipFile(BytesIO(content)) as archive:
            raw = archive.read("word/document.xml")
        root = ET.fromstring(raw)
    except Exception:
        return profile

    col_values: list[int] = []
    table_count = 0
    for node in root.iter():
        tag = str(node.tag)
        if tag.endswith("}cols"):
            for key, value in node.attrib.items():
                if key.endswith("}num") or key.endswith("num"):
                    try:
                        col_values.append(max(1, int(value)))
                    except Exception:
                        continue
        if tag.endswith("}tbl"):
            table_count += 1

    max_cols = max(col_values) if col_values else 1
    profile["column_count"] = _clamp_int(max_cols, default=1, min_value=1, max_value=4)
    profile["table_count"] = _clamp_int(profile.get("table_count", 0) + table_count, default=table_count, min_value=0, max_value=200)

    if max_cols >= 2:
        profile["detected_layout"] = "multi_column"
        profile["confidence"] = max(float(profile.get("confidence", 0.6)), 0.86)
        profile["signals"] = list(dict.fromkeys([*profile.get("signals", []), "docx_section_columns_detected"]))
    elif table_count >= 2:
        profile["detected_layout"] = "hybrid"
        profile["confidence"] = max(float(profile.get("confidence", 0.55)), 0.7)
        profile["signals"] = list(dict.fromkeys([*profile.get("signals", []), "docx_table_structure_detected"]))

    complexity = _clamp_int(
        int(profile.get("complexity_score", 20)) + min(18, table_count * 4) + (10 if max_cols >= 2 else 0),
        default=30,
        min_value=0,
        max_value=100,
    )
    profile["complexity_score"] = complexity
    profile["source_type"] = "word"
    profile["signals"] = profile.get("signals", [])[:20]
    return profile


def _extract_docx_text_fallback(content: bytes) -> tuple[str, int]:
    with ZipFile(BytesIO(content)) as archive:
        raw = archive.read("word/document.xml")
    root = ET.fromstring(raw)
    paragraphs: list[str] = []
    paragraph_count = 0
    for paragraph in root.iter():
        if not paragraph.tag.endswith("}p"):
            continue
        paragraph_count += 1
        texts: list[str] = []
        for node in paragraph.iter():
            if node.tag.endswith("}t") and node.text:
                value = node.text.strip()
                if value:
                    texts.append(value)
        if texts:
            paragraphs.append(" ".join(texts))
    return "\n".join(paragraphs), paragraph_count


def _extract_pptx_text_fallback(content: bytes) -> tuple[str, int]:
    with ZipFile(BytesIO(content)) as archive:
        slide_names = sorted(
            name for name in archive.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        )
        slides: list[str] = []
        for name in slide_names:
            raw = archive.read(name)
            root = ET.fromstring(raw)
            parts: list[str] = []
            for node in root.iter():
                if node.tag.endswith("}t") and node.text:
                    value = node.text.strip()
                    if value:
                        parts.append(value)
            if parts:
                slides.append(" ".join(parts))
    return "\n".join(slides), len(slide_names)


def _youtube_timedtext_transcript(video_id: str) -> str:
    try:
        import httpx

        with httpx.Client(timeout=12.0, follow_redirects=True) as client:
            track_list = client.get(
                "https://www.youtube.com/api/timedtext",
                params={"type": "list", "v": video_id},
            )
            if track_list.status_code >= 400 or not track_list.text.strip():
                return ""
            root = ET.fromstring(track_list.text.encode("utf-8"))
            track = None
            for candidate in root.iter("track"):
                lang_code = (candidate.attrib.get("lang_code") or "").lower()
                if lang_code.startswith("en"):
                    track = candidate
                    break
                if track is None:
                    track = candidate
            if track is None:
                return ""

            params: dict[str, str] = {"v": video_id, "lang": track.attrib.get("lang_code", "en")}
            kind = track.attrib.get("kind")
            if kind:
                params["kind"] = kind
            name = track.attrib.get("name")
            if name:
                params["name"] = name

            transcript_response = client.get("https://www.youtube.com/api/timedtext", params=params)
            if transcript_response.status_code >= 400 or not transcript_response.text.strip():
                return ""
            transcript_root = ET.fromstring(transcript_response.text.encode("utf-8"))
            lines: list[str] = []
            for node in transcript_root.iter("text"):
                if node.text:
                    cleaned = re.sub(r"\s+", " ", html.unescape(node.text)).strip()
                    if cleaned:
                        lines.append(cleaned)
            return " ".join(lines).strip()
    except Exception:
        return ""


def _youtube_metadata_text(url: str) -> str:
    video_id = _extract_youtube_id(url)
    if not video_id:
        return ""
    try:
        import httpx

        with httpx.Client(timeout=8.0, follow_redirects=True) as client:
            response = client.get(
                "https://www.youtube.com/oembed",
                params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            )
            if response.status_code >= 400:
                return f"YouTube URL: https://www.youtube.com/watch?v={video_id}."
            payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            title = _safe_str(payload.get("title"), max_len=220)
            author = _safe_str(payload.get("author_name"), max_len=120)
            if title and author:
                return (
                    f"Video title: {title}. Channel: {author}. "
                    "Transcript was unavailable, so summary quality may be limited."
                )
            if title:
                return f"Video title: {title}. Transcript was unavailable, so summary quality may be limited."
            return f"YouTube URL: https://www.youtube.com/watch?v={video_id}. Transcript was unavailable."
    except Exception:
        return f"YouTube URL: https://www.youtube.com/watch?v={video_id}. Transcript was unavailable."


def _youtube_transcript(url: str) -> str:
    video_id = _extract_youtube_id(url)
    if not video_id:
        return ""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        lines = YouTubeTranscriptApi.get_transcript(video_id)
        parsed = " ".join(item.get("text", "") for item in lines)
        parsed = re.sub(r"\s+", " ", parsed).strip()
        if parsed:
            return parsed
    except Exception:
        pass
    return _youtube_timedtext_transcript(video_id)


def extract_text_from_file(filename: str, content: bytes) -> ExtractTextResponse:
    ext = (filename.rsplit(".", 1)[-1].lower() if "." in filename else "")
    source_type = "text"
    details: dict[str, Any] = {"extension": ext}
    text = ""
    layout_profile: dict[str, Any] = _infer_layout_profile_from_text("", source_type="unknown")
    preview_meta: dict[str, Any] = {"renderer": "text", "display": "pre"}

    if ext in {"txt", "md", "rtf"}:
        source_type = "text"
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                text = content.decode(encoding)
                details["encoding"] = encoding
                break
            except UnicodeDecodeError:
                continue
        layout_profile = _infer_layout_profile_from_text(text, source_type="text")
        preview_meta = {"renderer": "text", "display": "pre", "mime_type": "text/plain"}
    elif ext == "pdf":
        source_type = "pdf"
        try:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(content))
            page_chunks: list[str] = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    page_chunks.append(page_text)
            text = "\n\n".join(page_chunks)
            details["pages"] = len(reader.pages)
            layout_profile = _detect_pdf_layout_profile(reader=reader, extracted_text=text)
            preview_meta = {"renderer": "pdf", "display": "iframe", "mime_type": "application/pdf"}
        except Exception as exc:
            raise ValueError("Unable to extract text from this PDF file.") from exc
    elif ext in {"docx", "doc"}:
        source_type = "word"
        if ext == "doc":
            raise ValueError("Legacy .doc is not supported. Convert to .docx.")
        try:
            try:
                from docx import Document

                doc = Document(BytesIO(content))
                text = "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
                details["paragraphs"] = len(doc.paragraphs)
                details["tables"] = len(doc.tables)
                details["parser"] = "python-docx"
            except Exception:
                text, paragraph_count = _extract_docx_text_fallback(content)
                details["paragraphs"] = paragraph_count
                details["parser"] = "zipxml-fallback"
            layout_profile = _detect_docx_layout_profile(content=content, extracted_text=text)
            preview_meta = {
                "renderer": "docx",
                "display": "docx-preview",
                "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            }
        except Exception as exc:
            raise ValueError("Unable to extract text from this Word document.") from exc
    elif ext in {"pptx", "ppt"}:
        source_type = "ppt"
        if ext == "ppt":
            raise ValueError("Legacy .ppt files are not supported. Please upload .pptx or paste text.")
        try:
            try:
                from pptx import Presentation

                prs = Presentation(BytesIO(content))
                chunks: list[str] = []
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if getattr(shape, "has_text_frame", False) and shape.text_frame:
                            value = shape.text_frame.text.strip()
                            if value:
                                chunks.append(value)
                text = "\n".join(chunks)
                details["slides"] = len(prs.slides)
                details["parser"] = "python-pptx"
            except Exception:
                text, slide_count = _extract_pptx_text_fallback(content)
                details["slides"] = slide_count
                details["parser"] = "zipxml-fallback"
            layout_profile = _infer_layout_profile_from_text(text, source_type="text")
            preview_meta = {"renderer": "text", "display": "pre", "mime_type": "text/plain"}
        except Exception as exc:
            raise ValueError("Unable to extract text from this PowerPoint file.") from exc
    elif ext in {"png", "jpg", "jpeg", "webp", "gif", "bmp"}:
        source_type = "image"
        details["note"] = "Image analysis uses OCR/vision extraction when available."
        try:
            from PIL import Image

            image = Image.open(BytesIO(content))
            details["width"] = image.width
            details["height"] = image.height
            details["parser"] = "pillow"
            text = ""
        except Exception:
            text = ""
        vision_text = vision_extract_text(content=content, filename=filename)
        if vision_text:
            text = vision_text
            details["generation_mode"] = "llm-vision"
        if not text:
            text = f"Image uploaded ({filename}). No text could be extracted automatically."
        layout_profile = _infer_layout_profile_from_text(text, source_type="image")
        preview_meta = {"renderer": "image", "display": "img", "mime_type": "image/*"}
    elif ext in {"mp4", "mov", "avi", "mkv", "webm", "m4v"}:
        source_type = "video"
        details["note"] = "Video transcription is best-effort. Add transcript notes if extraction fails."
        transcribed = transcribe_media(content=content, filename=filename)
        if transcribed:
            text = transcribed
            details["generation_mode"] = "llm-transcription"
        else:
            text = f"Video uploaded ({filename}). Please add transcript notes for higher-quality summary."
        layout_profile = _infer_layout_profile_from_text(text, source_type="unknown")
        preview_meta = {"renderer": "text", "display": "pre", "mime_type": "text/plain"}
    else:
        raise ValueError("Unsupported file type. Use .txt, .pdf, .docx, .pptx, image, or video files.")

    normalized = _normalize_extracted_text(text)
    if not normalized:
        raise ValueError("No extractable text was found in this file.")
    layout_profile["source_type"] = (
        "pdf"
        if source_type == "pdf"
        else "word"
        if source_type == "word"
        else "text"
        if source_type in {"text", "ppt"}
        else "image"
        if source_type == "image"
        else "unknown"
    )
    details["layout_profile"] = layout_profile
    details["preview_meta"] = preview_meta
    details["line_count"] = len(normalized.splitlines())
    details["character_count"] = len(normalized)

    return ExtractTextResponse(
        filename=filename,
        source_type=source_type,
        text=normalized,
        characters=len(normalized),
        details=details,
    )


def run_summarizer(payload: SummarizerRequest) -> SummarizerResponse:
    text = (payload.content or "").strip()
    metadata: dict[str, Any] = {"source_type": payload.source_type}

    if payload.source_url:
        metadata["source_url"] = payload.source_url

    if payload.source_type == "youtube" and payload.source_url and not text:
        text = _youtube_transcript(payload.source_url)
        metadata["transcript_fetched"] = bool(text)
        if not text:
            fallback_text = _youtube_metadata_text(payload.source_url)
            metadata["youtube_fallback_used"] = bool(fallback_text)
            if fallback_text:
                text = fallback_text

    if not text:
        if payload.source_type == "youtube":
            raise ValueError("Could not fetch YouTube transcript. Paste transcript text or upload notes as .txt.")
        raise ValueError("No content provided for summarization.")

    text = re.sub(r"\s+", " ", text).strip()
    sentences = _summarizer_sentences(text)
    summary_count, key_count, action_count = _length_to_limits(payload.length)

    if not sentences:
        sentences = [text]

    summary_sentences = sentences[:summary_count]
    summary = " ".join(summary_sentences)

    key_points = _best_sentences(sentences, key_count)
    action_items = [
        sentence
        for sentence in sentences
        if re.search(r"\b(should|must|need to|next|action|implement|update|fix|create|review)\b", sentence, re.IGNORECASE)
    ][:action_count]

    if not action_items:
        action_items = [f"Review: {item}" for item in key_points[:action_count]]
    generation_mode = "heuristic"

    llm_payload = json_completion(
        system_prompt=(
            "You are a high-precision summarization assistant. "
            "Return strict JSON only."
        ),
        user_prompt=(
            f"Language: {_locale_language_name(payload.locale)}.\n"
            f"Mode: {payload.mode}. Length: {payload.length}. Output language hint: {payload.output_language}.\n"
            "Use only provided content. Do not add external facts.\n"
            "Return JSON schema:\n"
            "{"
            "\"summary\": \"...\","
            "\"key_points\": [\"...\"],"
            "\"action_items\": [\"...\"]"
            "}\n"
            f"Rules: key_points max {key_count}, action_items max {action_count}. "
            "Action items must be concrete and imperative when possible.\n\n"
            f"Content:\n{text[:12000]}\n"
        ),
        temperature=0.15,
        max_output_tokens=1000,
    )
    if llm_payload:
        llm_summary = _safe_str(llm_payload.get("summary"), max_len=5000)
        llm_key_points = _safe_str_list(llm_payload.get("key_points"), max_items=key_count, max_len=280)
        llm_action_items = _safe_str_list(llm_payload.get("action_items"), max_items=action_count, max_len=280)
        if llm_summary and llm_key_points:
            summary = llm_summary
            key_points = llm_key_points
            if llm_action_items:
                action_items = llm_action_items
            generation_mode = "llm"

    if payload.mode == "key_points":
        summary = "\n".join(f"- {item}" for item in key_points)
    elif payload.mode == "action_items":
        summary = "\n".join(f"- {item}" for item in action_items)

    metadata["generation_mode"] = generation_mode

    return SummarizerResponse(
        summary=summary,
        key_points=key_points,
        action_items=action_items,
        word_count_in=len(text.split()),
        word_count_out=len(summary.split()),
        generated_at=datetime.now(timezone.utc),
        metadata=metadata,
    )


def run_additional_tool(payload: ToolRequest, tool_slug: str) -> ToolResponse:
    if tool_slug not in ADDITIONAL_TOOL_SLUGS:
        raise ValueError("Unsupported tool.")

    canonical_tool_slug = (
        "resume-optimization-report"
        if tool_slug in MERGED_INTO_RESUME_OPTIMIZATION_REPORT
        else tool_slug
    )

    base = _build_base_analysis(payload)
    locale = payload.locale
    top_skills = _important_terms(payload.resume_text, limit=12)
    tool_inputs = payload.tool_inputs if isinstance(payload.tool_inputs, dict) else {}
    target_role = _safe_str(tool_inputs.get("target_role"), max_len=120)
    country = _safe_str(tool_inputs.get("country"), max_len=120)
    years_input = _clamp_int(tool_inputs.get("years_experience"), default=_seniority_to_years(payload.candidate_profile.seniority), min_value=0, max_value=45)

    credibility = _credibility_score(payload.resume_text, payload.job_description_text)
    stuffing = _keyword_stuffing_report(payload.resume_text, base["matched_terms"] + base["missing_terms"])
    bullet_quality = _analyze_bullet_quality(payload.resume_text)
    humanization = _humanization_report(f"{payload.resume_text}\n{payload.job_description_text}")

    details: dict[str, Any] = {
        "tool": canonical_tool_slug,
        "requested_tool": tool_slug,
        "recommendation_label": _msg(locale, f"recommend_{base['recommendation']}"),
        "analysis_summary": base["analysis_summary"],
        "matched_term_evidence": base["matched_term_evidence"],
        "missing_term_context": base["missing_term_context"],
        "hard_filter_evidence": base["hard_filter_evidence"],
        "layout_analysis": base["layout_profile"],
        "layout_fit_for_target": base["layout_fit_for_target"],
        "format_recommendation": base["format_recommendation"],
        "resume_file_meta": base["resume_file_meta"],
    }
    generation_mode = _safe_str(base.get("generation_mode"), max_len=24).lower() or "heuristic"
    generation_scope = _safe_str(base.get("generation_scope"), max_len=40).lower() or "heuristic"

    if canonical_tool_slug == "resume-optimization-report":
        roi_gain = max(
            8,
            min(
                42,
                int(
                    round(
                        sum(item.impact_score for item in base["fix_plan"][:3])
                        / max(1, len(base["fix_plan"][:3]))
                        / 2.2
                    )
                ),
            ),
        )
        headline_terms = " | ".join(top_skills[:4])
        profile_summary = _safe_str(tool_inputs.get("profile_summary"), max_len=500)

        details["merged_features"] = [
            "one_click_optimize",
            "resume_score",
            "resume_summary_generator",
            "resume_bullet_points_generator",
            "ai_resume_tool",
            "linkedin_optimization_tool",
            "resume_builder_tool",
        ]
        details["score_breakdown"] = {
            "overall": int(round((base["scores"].job_match + base["scores"].ats_readability) / 2)),
            "job_match": base["scores"].job_match,
            "ats_readability": base["scores"].ats_readability,
            "credibility": credibility["score"],
        }
        details["optimized_actions"] = [
            {
                "title": item.title,
                "reason": item.reason,
                "estimated_minutes": item.effort_minutes,
            }
            for item in base["fix_plan"][:3]
        ]
        details["expected_match_gain"] = roi_gain
        details["report"] = {
            "high_risks": [risk.message for risk in base["risks"] if risk.severity == "high"],
            "top_fixes": [item.title for item in base["fix_plan"][:5]],
        }
        details["generated_summary"] = _safe_str(
            f"Target role focus: {target_role or 'role not specified'}. "
            f"Primary matched terms: {', '.join(base['matched_terms'][:6]) or 'none detected'}. "
            f"Top gaps: {', '.join(base['missing_terms'][:4]) or 'no major term gaps'}."
        )

        evidence_bullets: list[str] = []
        for snippets in (base.get("matched_term_evidence") or {}).values():
            if not isinstance(snippets, list):
                continue
            for snippet in snippets:
                clean = _safe_str(snippet, max_len=220)
                if clean and clean not in evidence_bullets:
                    evidence_bullets.append(clean)
                if len(evidence_bullets) >= 6:
                    break
            if len(evidence_bullets) >= 6:
                break
        details["generated_bullets"] = evidence_bullets or [item.reason for item in base["fix_plan"][:4]]

        ai_suggestions = [f"{item.title}: {item.reason}" for item in base["fix_plan"][:5]]
        details["ai_suggestions"] = ai_suggestions
        details["linkedin_suggestions"] = {
            "headline": f"{payload.candidate_profile.seniority.title()} Engineer | {target_role or headline_terms}",
            "about": "Focus on measurable outcomes, scope, and core tools used in production.",
        }
        details["builder_sections"] = [
            "Header",
            "Professional Summary",
            "Skills",
            "Experience",
            "Projects",
            "Education",
        ]
        details["starter_summary"] = (
            profile_summary
            or f"{payload.candidate_profile.seniority.title()} engineer with strengths in {', '.join(top_skills[:5])}."
        )

        llm_payload = json_completion(
            system_prompt=(
                "You are a resume optimization assistant. "
                "Return strict JSON only."
            ),
            user_prompt=(
                f"Language: {_locale_language_name(locale)}.\n"
                "Produce merged outputs for a resume optimization report.\n"
                "Use resume and JD evidence only. No fake claims.\n"
                "Return JSON schema:\n"
                "{"
                "\"generated_summary\":\"...\","
                "\"generated_bullets\":[\"...\"],"
                "\"ai_suggestions\":[\"...\"],"
                "\"linkedin_suggestions\":{\"headline\":\"...\",\"about\":\"...\"},"
                "\"starter_summary\":\"...\""
                "}\n"
                "Rules: generated_bullets 4-6 items, ai_suggestions exactly 5, concise and practical.\n\n"
                f"Resume:\n{payload.resume_text[:4200]}\n"
                f"Job description:\n{payload.job_description_text[:3000]}\n"
                f"Target role: {target_role or 'not provided'}\n"
                f"Matched terms: {', '.join(base['matched_terms'][:12])}\n"
                f"Missing terms: {', '.join(base['missing_terms'][:10])}\n"
                f"Top risks: {', '.join([f'{r.type}:{r.severity}' for r in base['risks'][:5]])}\n"
            ),
            temperature=0.2,
            max_output_tokens=900,
        )
        if llm_payload:
            llm_summary = _safe_str(llm_payload.get("generated_summary"), max_len=800)
            llm_bullets = _safe_str_list(llm_payload.get("generated_bullets"), max_items=6, max_len=260)
            llm_suggestions = _safe_str_list(llm_payload.get("ai_suggestions"), max_items=5, max_len=260)
            llm_starter_summary = _safe_str(llm_payload.get("starter_summary"), max_len=800)
            llm_linkedin = llm_payload.get("linkedin_suggestions")

            if llm_summary:
                details["generated_summary"] = llm_summary
            if llm_bullets:
                details["generated_bullets"] = llm_bullets
            if llm_suggestions:
                details["ai_suggestions"] = llm_suggestions
            if llm_starter_summary:
                details["starter_summary"] = llm_starter_summary
            if isinstance(llm_linkedin, dict):
                llm_headline = _safe_str(llm_linkedin.get("headline"), max_len=180)
                llm_about = _safe_str(llm_linkedin.get("about"), max_len=420)
                if llm_headline or llm_about:
                    details["linkedin_suggestions"] = {
                        "headline": llm_headline or details["linkedin_suggestions"]["headline"],
                        "about": llm_about or details["linkedin_suggestions"]["about"],
                    }

            generation_mode = "llm"
            generation_scope = "merged-report"

        details["bullet_quality_analyzer"] = bullet_quality
        details["keyword_stuffing_detector"] = stuffing
        details["resume_credibility_score"] = credibility
        details["humanization_filter"] = _humanization_report(
            f"{_safe_str(details.get('generated_summary'), max_len=1200)}\n"
            f"{_safe_str(details.get('starter_summary'), max_len=1200)}"
        )
    elif canonical_tool_slug == "job-application-tracker":
        applied = _clamp_int(tool_inputs.get("applied"), default=0, min_value=0, max_value=10000)
        interviews = _clamp_int(tool_inputs.get("interviews"), default=0, min_value=0, max_value=10000)
        offers = _clamp_int(tool_inputs.get("offers"), default=0, min_value=0, max_value=10000)
        interview_rate = round((interviews / applied) * 100, 1) if applied else 0.0
        offer_rate = round((offers / max(interviews, 1)) * 100, 1) if interviews else 0.0
        details["tracker_template"] = {
            "columns": ["Company", "Role", "Date Applied", "Stage", "Next Action", "Notes"],
            "recommended_next_action": "Set one follow-up reminder 5 business days after application.",
            "applied": applied,
            "interviews": interviews,
            "offers": offers,
            "interview_rate_percent": interview_rate,
            "offer_rate_percent": offer_rate,
        }
    elif canonical_tool_slug == "jobs":
        work_mode = _safe_str(tool_inputs.get("work_mode"), max_len=30).lower() or "remote"
        details["job_search_queries"] = [f"{term} engineer {work_mode} {country}".strip() for term in (top_skills[:4] or ["software"])]
        details["target_country"] = country or "global"
    elif canonical_tool_slug == "career-change-tool":
        target_track = _safe_str(tool_inputs.get("target_track"), max_len=120) or target_role
        details["career_change"] = {
            "transferable_skills": top_skills[:8],
            "bridge_actions": [
                "Add 2 role-aligned portfolio projects.",
                "Map previous impact to new domain outcomes.",
                "Tailor summary and headline to target track.",
            ],
            "target_track": target_track or "selected career track",
        }
    elif canonical_tool_slug == "product-walkthrough":
        details["walkthrough"] = [
            "Start with Resume Score for baseline.",
            "Use One-Click Optimize for minimum edits.",
            "Run Job Match before each application.",
            "Generate role-specific cover letter and interview prep.",
        ]
    elif canonical_tool_slug == "job-application-roi-calculator":
        minutes_per_application = _clamp_int(tool_inputs.get("minutes_per_application"), default=35, min_value=5, max_value=300)
        expected_match = _clamp_int(base["scores"].job_match, default=60, min_value=1, max_value=100)
        expected_interview_probability = round(max(0.05, min(0.85, (expected_match / 120) + (credibility["score"] / 350))), 2)
        roi = round(((expected_interview_probability * 100) / max(1, minutes_per_application)) * 10, 1)
        details["roi"] = {
            "minutes_per_application": minutes_per_application,
            "expected_interview_probability": expected_interview_probability,
            "roi_score": roi,
            "recommendation": "skip" if roi < 8 else "fix" if roi < 18 else "apply",
        }
    elif canonical_tool_slug == "seniority-calibration-tool":
        required_years = _clamp_int(tool_inputs.get("required_years"), default=max((int(x) for x in YEARS_RE.findall(payload.job_description_text.lower())), default=0), min_value=0, max_value=40)
        gap = required_years - years_input
        details["seniority_calibration"] = {
            "candidate_years_signal": years_input,
            "required_years_signal": required_years,
            "gap_years": gap,
            "classification": "underqualified" if gap > 1 else "overqualified" if gap < -3 else "aligned",
            "actions": [
                "Adjust headline to align level with role scope.",
                "Show scope and ownership in 2 recent projects.",
            ],
        }
    elif canonical_tool_slug == "rejection-reason-classifier":
        ranked = sorted(base["risks"], key=lambda risk: (1 if risk.severity == "high" else 0), reverse=True)
        details["rejection_reasons"] = [
            {"type": risk.type, "severity": risk.severity, "reason": risk.message}
            for risk in ranked[:4]
        ]
        details["top_likely_rejection"] = ranked[0].type if ranked else "keyword_gap"
    elif canonical_tool_slug == "cv-region-translator":
        region_mode = _safe_str(tool_inputs.get("region_mode"), max_len=20).lower() or "eu"
        details["region_translation"] = {
            "mode": region_mode,
            "format_rules": [
                "Use reverse-chronological experience.",
                "Keep profile concise and evidence-backed.",
                "Match role titles to local market wording.",
            ],
            "required_adaptations": [
                "Adjust date format and section naming.",
                "Tune summary tone for regional expectations.",
            ],
        }

    # Always provide these core quality signals for resume/career tools.
    details["resume_credibility_score"] = details.get("resume_credibility_score", credibility)
    details["keyword_stuffing_detector"] = details.get("keyword_stuffing_detector", stuffing)
    details["humanization_filter"] = details.get("humanization_filter", humanization)
    ai_insights = _additional_ai_insights(
        tool_slug=canonical_tool_slug,
        locale=locale,
        resume_text=payload.resume_text,
        job_description_text=payload.job_description_text,
        tool_inputs=tool_inputs,
        risks=base["risks"],
        fix_plan=base["fix_plan"],
    )
    if ai_insights:
        details["ai_insights"] = ai_insights
        generation_mode = "llm"
        if generation_scope == "heuristic":
            generation_scope = "insight-enrichment"
    details["generation_mode"] = generation_mode
    details["generation_scope"] = generation_scope

    quality_texts: list[str] = [
        _safe_str(details.get("analysis_summary"), max_len=400),
        _safe_str(details.get("generated_summary"), max_len=400),
        _safe_str(details.get("starter_summary"), max_len=400),
    ]
    generated_bullets = details.get("generated_bullets")
    if isinstance(generated_bullets, list):
        quality_texts.extend(_safe_str_list(generated_bullets, max_items=4, max_len=220))
    if isinstance(details.get("ai_suggestions"), list):
        quality_texts.extend(_safe_str_list(details.get("ai_suggestions"), max_items=4, max_len=220))

    _ensure_quality_generation(
        tool_slug=canonical_tool_slug,
        generation_mode=generation_mode,
        generation_scope=generation_scope,
        sample_texts=[text for text in quality_texts if text],
    )

    return ToolResponse(
        recommendation=base["recommendation"],
        confidence=base["confidence"],
        scores=base["scores"],
        risks=base["risks"],
        fix_plan=base["fix_plan"],
        generated_at=datetime.now(timezone.utc),
        details=details,
    )


VPN_TOOL_SLUGS = {
    "is-my-vpn-working",
    "ip-dns-webrtc-leak-tester",
    "vpn-country-compatibility-checker",
    "find-best-vpn-for-me-quiz",
    "vpn-speed-expectation-calculator",
    "vpn-block-detection-tool",
}

COUNTRY_COMPATIBILITY: dict[str, dict[str, str]] = {
    "uae": {
        "legal": "Restricted for unlicensed usage.",
        "streaming": "Usually works with obfuscated servers.",
        "banking": "Generally works but can trigger additional verification.",
        "risk": "high",
    },
    "china": {
        "legal": "Highly restricted and heavily enforced.",
        "streaming": "Limited without advanced obfuscation.",
        "banking": "Often unstable from foreign endpoints.",
        "risk": "high",
    },
    "turkey": {
        "legal": "Legal but selective throttling can occur.",
        "streaming": "Usually works.",
        "banking": "Works in most cases.",
        "risk": "medium",
    },
    "pakistan": {
        "legal": "Allowed with periodic restrictions.",
        "streaming": "Works for many providers.",
        "banking": "Usually works with local server fallback.",
        "risk": "medium",
    },
    "india": {
        "legal": "Legal with logging policy considerations.",
        "streaming": "Usually works.",
        "banking": "Works with major providers.",
        "risk": "medium",
    },
}

VPN_PROVIDER_CATALOG: list[dict[str, Any]] = [
    {
        "provider": "NordVPN",
        "strengths": {"streaming", "speed", "privacy", "gaming", "obfuscation"},
        "budget": "medium",
        "device_limit": 10,
        "banking": "strong",
        "obfuscation": True,
        "edge": "Large global network with strong performance consistency.",
    },
    {
        "provider": "Surfshark",
        "strengths": {"budget", "streaming", "privacy", "family"},
        "budget": "low",
        "device_limit": 100,
        "banking": "good",
        "obfuscation": True,
        "edge": "Good value profile for many devices and shared households.",
    },
    {
        "provider": "Proton VPN",
        "strengths": {"privacy", "security", "compliance", "research"},
        "budget": "medium",
        "device_limit": 10,
        "banking": "good",
        "obfuscation": True,
        "edge": "Strong security and transparency posture for privacy-focused users.",
    },
    {
        "provider": "ExpressVPN",
        "strengths": {"streaming", "travel", "usability", "reliability"},
        "budget": "high",
        "device_limit": 8,
        "banking": "strong",
        "obfuscation": True,
        "edge": "Reliable apps and stable cross-region connectivity.",
    },
    {
        "provider": "Mullvad",
        "strengths": {"privacy", "security", "anonymity"},
        "budget": "medium",
        "device_limit": 5,
        "banking": "moderate",
        "obfuscation": False,
        "edge": "Minimal-account model and strong privacy defaults.",
    },
    {
        "provider": "Windscribe",
        "strengths": {"budget", "privacy", "streaming", "light-use"},
        "budget": "low",
        "device_limit": 12,
        "banking": "moderate",
        "obfuscation": True,
        "edge": "Flexible plans for lighter usage and cost control.",
    },
]


def _budget_alignment_score(target_budget: str, provider_budget: str) -> int:
    if target_budget == provider_budget:
        return 8
    if target_budget == "low" and provider_budget == "high":
        return -8
    if target_budget == "high" and provider_budget == "low":
        return 4
    return 1


def _provider_fit_recommendations(
    *,
    country: str,
    use_case: str,
    budget: str,
    devices: int,
) -> tuple[list[dict[str, Any]], str]:
    country_data = COUNTRY_COMPATIBILITY.get(country.lower(), {"risk": "medium"})
    country_risk = _safe_str(country_data.get("risk"), max_len=12).lower() or "medium"

    ranked: list[dict[str, Any]] = []
    for provider in VPN_PROVIDER_CATALOG:
        p_name = provider["provider"]
        strengths = {str(item).lower() for item in provider.get("strengths", set())}
        p_edge = _safe_str(provider.get("edge"), max_len=120)
        p_budget = _safe_str(provider.get("budget"), max_len=16).lower() or "medium"
        p_devices = _clamp_int(provider.get("device_limit"), default=5, min_value=1, max_value=200)
        p_obfuscation = bool(provider.get("obfuscation"))
        p_banking = _safe_str(provider.get("banking"), max_len=16).lower() or "moderate"

        score = 46
        reasons: list[str] = []
        if p_edge:
            reasons.append(p_edge)

        if use_case in strengths:
            score += 15
            reasons.append(f"Strong fit for {use_case} workloads.")
        elif use_case == "work" and ("reliability" in strengths or "security" in strengths):
            score += 10
            reasons.append("Good reliability and security profile for work traffic.")
        elif use_case == "banking" and p_banking in {"good", "strong"}:
            score += 10
            reasons.append("Good stability profile for banking and verification flows.")
        else:
            score += 2

        budget_score = _budget_alignment_score(budget, p_budget)
        score += budget_score
        if budget_score > 5:
            reasons.append("Budget alignment is favorable.")
        elif budget_score < 0:
            reasons.append("Price tier may be above your selected budget.")

        if devices <= p_devices:
            score += 6
            if devices >= 5:
                reasons.append("Device allowance fits multi-device usage.")
        elif devices > p_devices + 3:
            score -= 10
            reasons.append("Device allowance may be restrictive for your setup.")
        else:
            score -= 3

        if country_risk == "high":
            if p_obfuscation:
                score += 8
                reasons.append("Obfuscation support helps in restrictive networks.")
            else:
                score -= 10
                reasons.append("Limited anti-blocking support for restrictive regions.")
        elif country_risk == "medium" and p_obfuscation:
            score += 4

        if use_case == "streaming" and ("streaming" in strengths or "travel" in strengths):
            score += 5
        if use_case == "privacy" and ("privacy" in strengths or "anonymity" in strengths):
            score += 5

        fit_score = _clamp_int(score, default=70, min_value=1, max_value=100)
        caution = ""
        if budget == "low" and p_budget == "high":
            caution = "May be expensive versus your budget target."
        elif devices > p_devices:
            caution = "Check device-limit policy before purchase."
        elif country_risk == "high" and not p_obfuscation:
            caution = "Not ideal for strict blocking environments."

        ranked.append(
            {
                "provider": p_name,
                "fit_score": fit_score,
                "reason": " ".join(reasons[:2]) or "Balanced profile for your inputs.",
                "best_for": use_case.title(),
                "caution": caution,
            }
        )

    ranked.sort(key=lambda item: int(item.get("fit_score", 0)), reverse=True)
    return ranked[:3], country_risk


def run_vpn_probe_enrich(payload: VpnProbeEnrichRequest) -> VpnProbeEnrichResponse:
    all_ips = _extract_ip_list(payload.webrtc_ips) + _extract_ip_list(payload.baseline_webrtc_ips)
    all_ips += _extract_ip_list(payload.dns_resolver_ips)
    all_ips += _extract_ip_list([payload.public_ip] if payload.public_ip else [])
    all_ips += _extract_ip_list([payload.baseline_public_ip] if payload.baseline_public_ip else [])

    deduped: list[str] = []
    seen: set[str] = set()
    for ip_value in all_ips:
        if ip_value in seen:
            continue
        seen.add(ip_value)
        deduped.append(ip_value)

    records: list[VpnProbeGeoRecord] = []
    mapping: dict[str, dict[str, Any]] = {}
    for ip_value in deduped:
        geo = _geo_lookup_ip(ip_value)
        mapping[ip_value] = geo
        records.append(
            VpnProbeGeoRecord(
                ip=ip_value,
                country=_safe_str(geo.get("country"), max_len=120) or None,
                country_code=_safe_str(geo.get("country_code"), max_len=10) or None,
                region=_safe_str(geo.get("region"), max_len=120) or None,
                city=_safe_str(geo.get("city"), max_len=120) or None,
                isp=_safe_str(geo.get("isp"), max_len=180) or None,
                is_private=bool(geo.get("is_private")),
                source=_safe_str(geo.get("source"), max_len=40) or "unknown",
            )
        )

    current_country = _safe_str(mapping.get(payload.public_ip or "", {}).get("country"), max_len=120) if payload.public_ip else ""
    baseline_country = _safe_str(mapping.get(payload.baseline_public_ip or "", {}).get("country"), max_len=120) if payload.baseline_public_ip else ""
    dns_records = [mapping.get(ip) or {} for ip in _extract_ip_list(payload.dns_resolver_ips)]
    webrtc_records = [mapping.get(ip) or {} for ip in _extract_ip_list(payload.webrtc_ips)]
    baseline_webrtc_records = [mapping.get(ip) or {} for ip in _extract_ip_list(payload.baseline_webrtc_ips)]

    summary = {
        "public_ip": payload.public_ip or "",
        "public_country": current_country,
        "baseline_public_ip": payload.baseline_public_ip or "",
        "baseline_country": baseline_country,
        "dns_countries": sorted({item.get("country") for item in dns_records if item.get("country")}),
        "webrtc_countries": sorted({item.get("country") for item in webrtc_records if item.get("country")}),
        "baseline_webrtc_countries": sorted({item.get("country") for item in baseline_webrtc_records if item.get("country")}),
        "expected_country": _safe_str(payload.expected_country, max_len=120),
    }

    return VpnProbeEnrichResponse(
        records=records,
        summary=summary,
        generated_at=datetime.now(timezone.utc),
    )


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _verdict_from_score(score: int) -> VpnToolVerdict:
    if score >= 75:
        return "good"
    if score >= 45:
        return "attention"
    return "critical"


def _build_is_my_vpn_working(payload: VpnToolRequest) -> VpnToolResponse:
    source = payload.input
    expected_country = _norm_text(source.get("expected_country"))

    baseline_public_ip = _norm_text(source.get("baseline_public_ip"))
    current_public_ip = _norm_text(source.get("current_public_ip")) or _norm_text(source.get("public_ip"))
    detected_country = _norm_text(source.get("current_country")) or _norm_text(source.get("detected_country"))
    baseline_country = _norm_text(source.get("baseline_country"))

    current_webrtc_ips = _extract_ip_list(source.get("current_webrtc_ips") or source.get("webrtc_ips"))
    baseline_webrtc_ips = _extract_ip_list(source.get("baseline_webrtc_ips"))
    dns_resolver_ips = _extract_ip_list(source.get("dns_resolver_ips"))

    # Backward compatibility for old manual mode.
    ip_changed = _as_bool(source.get("ip_changed"), default=bool(baseline_public_ip and current_public_ip and baseline_public_ip != current_public_ip))
    dns_leak_manual = _as_bool(source.get("dns_leak"), default=False)
    webrtc_leak_manual = _as_bool(source.get("webrtc_leak"), default=False)

    enriched = run_vpn_probe_enrich(
        VpnProbeEnrichRequest(
            locale=payload.locale,
            session_id=payload.session_id,
            public_ip=current_public_ip or None,
            baseline_public_ip=baseline_public_ip or None,
            webrtc_ips=current_webrtc_ips,
            baseline_webrtc_ips=baseline_webrtc_ips,
            dns_resolver_ips=dns_resolver_ips,
            expected_country=expected_country or None,
        )
    )

    record_map = {record.ip: record for record in enriched.records}
    if not detected_country and current_public_ip and current_public_ip in record_map:
        detected_country = record_map[current_public_ip].country or ""
    if not baseline_country and baseline_public_ip and baseline_public_ip in record_map:
        baseline_country = record_map[baseline_public_ip].country or ""

    dns_countries = sorted(
        {
            record_map[ip_value].country
            for ip_value in dns_resolver_ips
            if ip_value in record_map and record_map[ip_value].country
        }
    )
    current_webrtc_public = [ip_value for ip_value in current_webrtc_ips if _is_public_ip(ip_value)]
    webrtc_countries = sorted(
        {
            record_map[ip_value].country
            for ip_value in current_webrtc_public
            if ip_value in record_map and record_map[ip_value].country
        }
    )

    dns_leak = dns_leak_manual
    if dns_countries and expected_country:
        dns_leak = any(not _country_match(expected_country, country) for country in dns_countries if country)
    elif dns_resolver_ips and not dns_countries:
        dns_leak = dns_leak_manual

    webrtc_leak = webrtc_leak_manual
    if baseline_public_ip and current_webrtc_public and baseline_public_ip in current_webrtc_public:
        webrtc_leak = True
    if expected_country and webrtc_countries:
        if any(not _country_match(expected_country, country) for country in webrtc_countries if country):
            webrtc_leak = True
    if detected_country and webrtc_countries:
        if any(not _country_match(detected_country, country) for country in webrtc_countries if country):
            webrtc_leak = True

    score = 100
    cards: list[VpnToolCard] = []

    cards.append(
        VpnToolCard(
            title="IP changed from original connection",
            status="pass" if ip_changed else "fail",
            value="Yes" if ip_changed else "No",
            detail=(
                f"Baseline {baseline_public_ip} vs current {current_public_ip}."
                if baseline_public_ip and current_public_ip
                else ("VPN likely not tunneling traffic." if not ip_changed else "Primary tunnel signal looks healthy.")
            ),
        )
    )
    if not ip_changed:
        score -= 40

    country_matches = not expected_country or not detected_country or _country_match(expected_country, detected_country)
    cards.append(
        VpnToolCard(
            title="Expected country match",
            status="pass" if country_matches else "warn",
            value=f"{detected_country or 'Unknown'}",
            detail=f"Expected {expected_country}." if expected_country else "No expected country provided.",
        )
    )
    if not country_matches:
        score -= 15

    cards.append(
        VpnToolCard(
            title="DNS leak check",
            status="fail" if dns_leak else "pass",
            value="Leak detected" if dns_leak else "No leak detected",
            detail=(
                f"Resolver countries: {', '.join(dns_countries)}."
                if dns_countries
                else ("DNS requests appear exposed to non-VPN resolvers." if dns_leak else "DNS appears routed through VPN path.")
            ),
        )
    )
    if dns_leak:
        score -= 25

    cards.append(
        VpnToolCard(
            title="WebRTC leak check",
            status="warn" if webrtc_leak else "pass",
            value="Leak detected" if webrtc_leak else "No leak detected",
            detail=(
                f"WebRTC countries: {', '.join(webrtc_countries)}."
                if webrtc_countries
                else ("Browser may expose local/public IP via WebRTC." if webrtc_leak else "No WebRTC leakage signal.")
            ),
        )
    )
    if webrtc_leak:
        score -= 15

    score = max(0, min(100, score))
    verdict = _verdict_from_score(score)
    actions = [
        "Switch to a nearby VPN server and retest.",
        "Enable DNS leak protection in your VPN app.",
        "Disable WebRTC or use browser extension hardening.",
    ]
    headline = "VPN tunnel looks healthy." if verdict == "good" else "VPN setup needs attention before sensitive use."

    return VpnToolResponse(
        tool="is-my-vpn-working",
        headline=headline,
        verdict=verdict,
        score=score,
        cards=cards,
        actions=actions,
        details={
            "expected_country": expected_country,
            "detected_country": detected_country,
            "baseline_country": baseline_country,
            "baseline_public_ip": baseline_public_ip,
            "current_public_ip": current_public_ip,
            "dns_countries": dns_countries,
            "webrtc_countries": webrtc_countries,
            "probe_records": [record.model_dump() for record in enriched.records],
            "checks_passed": sum(1 for card in cards if card.status == "pass"),
        },
        generated_at=datetime.now(timezone.utc),
    )


def _build_leak_tester(payload: VpnToolRequest) -> VpnToolResponse:
    source = payload.input
    vpn_country = _norm_text(source.get("vpn_country"))
    public_ip = _norm_text(source.get("public_ip"))
    ip_country = _norm_text(source.get("ip_country")) or _norm_text(source.get("public_country"))
    dns_country = _norm_text(source.get("dns_country"))
    webrtc_country = _norm_text(source.get("webrtc_country"))
    dns_resolver_ips = _extract_ip_list(source.get("dns_resolver_ips"))
    webrtc_ips = _extract_ip_list(source.get("webrtc_ips"))

    enriched = run_vpn_probe_enrich(
        VpnProbeEnrichRequest(
            locale=payload.locale,
            session_id=payload.session_id,
            public_ip=public_ip or None,
            webrtc_ips=webrtc_ips,
            dns_resolver_ips=dns_resolver_ips,
            expected_country=vpn_country or None,
        )
    )
    record_map = {record.ip: record for record in enriched.records}
    if not ip_country and public_ip and public_ip in record_map:
        ip_country = record_map[public_ip].country or ""

    if not dns_country and dns_resolver_ips:
        dns_country = _dominant_country([record_map[ip_value].model_dump() for ip_value in dns_resolver_ips if ip_value in record_map]) or ""
    if not webrtc_country and webrtc_ips:
        webrtc_country = _dominant_country([record_map[ip_value].model_dump() for ip_value in webrtc_ips if ip_value in record_map]) or ""

    score = 100
    cards: list[VpnToolCard] = []
    leaks: list[str] = []

    def add_country_check(title: str, country: str):
        nonlocal score
        if not vpn_country or not country:
            cards.append(VpnToolCard(title=title, status="info", value=country or "Unknown", detail="Insufficient data for validation."))
            return
        if _country_match(vpn_country, country):
            cards.append(VpnToolCard(title=title, status="pass", value=country, detail="Country matches selected VPN region."))
            return
        cards.append(VpnToolCard(title=title, status="fail", value=country, detail=f"Expected {vpn_country}."))
        leaks.append(title)
        score -= 28

    add_country_check("Public IP location", ip_country)
    add_country_check("DNS resolver location", dns_country)
    add_country_check("WebRTC exposed location", webrtc_country)

    score = max(0, min(100, score))
    verdict = _verdict_from_score(score)
    headline = "No strong leak signal detected." if verdict == "good" else "Leak risk detected in one or more paths."
    actions = [
        "Enable kill switch and leak protection options.",
        "Force secure DNS in the VPN client.",
        "Retest in private window after reconnecting VPN.",
    ]
    if leaks:
        actions.insert(0, f"Investigate mismatch in: {', '.join(leaks)}.")

    return VpnToolResponse(
        tool="ip-dns-webrtc-leak-tester",
        headline=headline,
        verdict=verdict,
        score=score,
        cards=cards,
        actions=actions,
        details={
            "vpn_country": vpn_country,
            "public_ip": public_ip,
            "leaks": leaks,
            "probe_records": [record.model_dump() for record in enriched.records],
        },
        generated_at=datetime.now(timezone.utc),
    )


def _build_country_checker(payload: VpnToolRequest) -> VpnToolResponse:
    source = payload.input
    country = _norm_text(source.get("country"))
    use_case = _norm_text(source.get("use_case")) or "general"
    data = COUNTRY_COMPATIBILITY.get(country.lower(), {
        "legal": "No specific restriction data found. Verify local laws before usage.",
        "streaming": "Likely to work with major providers.",
        "banking": "May require local-region fallback server.",
        "risk": "medium",
    })

    risk = data["risk"]
    score = 80 if risk == "low" else 62 if risk == "medium" else 38
    verdict = _verdict_from_score(score)
    cards = [
        VpnToolCard(title="Legal status", status="fail" if risk == "high" else "warn" if risk == "medium" else "pass", value=data["legal"]),
        VpnToolCard(title="Streaming compatibility", status="warn" if risk == "high" else "pass", value=data["streaming"]),
        VpnToolCard(title="Banking compatibility", status="warn" if risk != "low" else "pass", value=data["banking"]),
        VpnToolCard(title="Target use case", status="info", value=use_case.title(), detail="Use case can change server and protocol recommendation."),
    ]
    actions = [
        "Use obfuscated servers where restrictions are high.",
        "Keep one local-region profile for banking apps.",
        "Retest important services after each VPN protocol switch.",
    ]

    return VpnToolResponse(
        tool="vpn-country-compatibility-checker",
        headline=f"Compatibility overview for {country or 'selected country'}.",
        verdict=verdict,
        score=score,
        cards=cards,
        actions=actions,
        details={"country": country, "risk_level": risk, "use_case": use_case},
        generated_at=datetime.now(timezone.utc),
    )


def _build_best_vpn_quiz(payload: VpnToolRequest) -> VpnToolResponse:
    source = payload.input
    country = _norm_text(source.get("country"))
    use_case = _norm_text(source.get("use_case")).lower() or "privacy"
    budget = _norm_text(source.get("budget")).lower() or "medium"
    devices = int(_as_float(source.get("devices"), 1))
    devices = max(1, devices)
    if use_case not in {"privacy", "streaming", "banking", "work"}:
        use_case = "privacy"
    if budget not in {"low", "medium", "high"}:
        budget = "medium"

    recommendations, country_risk = _provider_fit_recommendations(
        country=country,
        use_case=use_case,
        budget=budget,
        devices=devices,
    )
    avg_fit = int(round(sum(int(item.get("fit_score", 70)) for item in recommendations) / max(1, len(recommendations))))
    score = _clamp_int(avg_fit, default=74, min_value=1, max_value=100)
    verdict = _verdict_from_score(score)
    headline = "Personalized VPN shortlist generated."

    cards = [
        VpnToolCard(title="Primary use case", status="info", value=use_case.title()),
        VpnToolCard(title="Budget profile", status="info", value=budget.title()),
        VpnToolCard(title="Device count", status="info", value=str(devices)),
        VpnToolCard(title="Country context", status="info", value=country or "Not specified"),
    ]
    if country_risk == "high":
        cards.append(
            VpnToolCard(
                title="Restriction profile",
                status="warn",
                value="High",
                detail="Prefer providers with obfuscation and fallback protocol options.",
            )
        )
    actions = [
        f"Shortlist {item['provider']} first (fit {item['fit_score']}/100)."
        for item in recommendations[:3]
    ]
    actions.append("Verify current pricing and local policy requirements before purchase.")
    llm_explanation_text = ""

    generation_mode = "heuristic"
    generation_scope = "heuristic"
    llm_payload = json_completion(
        system_prompt=(
            "You are a practical VPN recommendation assistant. "
            "Return strict JSON only."
        ),
        user_prompt=(
            f"Language: {_locale_language_name(payload.locale)}.\n"
            "Generate recommendation output for a VPN quiz.\n"
            "Use only this provider list unless no fit exists: "
            f"{[item['provider'] for item in recommendations]}.\n"
            "Keep legal wording cautious and non-definitive.\n"
            "Return JSON schema:\n"
            "{"
            "\"headline\":\"...\","
            "\"score\":0,"
            "\"verdict\":\"good|attention|critical\","
            "\"recommendations\":["
            "{\"provider\":\"...\",\"reason\":\"...\",\"best_for\":\"...\",\"caution\":\"...\",\"fit_score\":0}"
            "],"
            "\"actions\":[\"...\"],"
            "\"llm_explanation\":\"...\""
            "}\n"
            "Rules: recommendations must be 2-4 items with specific reasons tied to user inputs. "
            "Do not claim guaranteed outcomes.\n\n"
            f"User input: country={country or 'unknown'}, use_case={use_case}, budget={budget}, devices={devices}\n"
            f"Country risk: {country_risk}\n"
            f"Heuristic recommendations: {recommendations}\n"
        ),
        temperature=0.25,
        max_output_tokens=750,
    )
    if llm_payload:
        llm_headline = _safe_str(llm_payload.get("headline"), max_len=180)
        llm_recommendations = _safe_vpn_recommendations(llm_payload.get("recommendations"), max_items=4)
        llm_actions = _safe_str_list(llm_payload.get("actions"), max_items=5, max_len=200)
        llm_explanation = _safe_str(llm_payload.get("llm_explanation"), max_len=500)
        llm_score = _clamp_int(llm_payload.get("score"), default=score, min_value=1, max_value=100)
        llm_verdict_raw = _safe_str(llm_payload.get("verdict"), max_len=20).lower()
        llm_verdict: VpnToolVerdict = verdict
        if llm_verdict_raw in {"good", "attention", "critical"}:
            llm_verdict = llm_verdict_raw  # type: ignore[assignment]

        if llm_recommendations:
            recommendations = llm_recommendations
            generation_mode = "llm"
            generation_scope = "full-analysis"
            score = llm_score
            verdict = llm_verdict
        if llm_headline:
            headline = llm_headline
        if llm_actions:
            actions = llm_actions
        if llm_explanation:
            llm_explanation_text = llm_explanation

    return VpnToolResponse(
        tool="find-best-vpn-for-me-quiz",
        headline=headline,
        verdict=verdict,
        score=score,
        cards=cards,
        actions=actions,
        details={
            "recommendations": recommendations,
            "country": country,
            "country_risk": country_risk,
            "use_case": use_case,
            "budget": budget,
            "devices": devices,
            "generation_mode": generation_mode,
            "generation_scope": generation_scope,
            "llm_explanation": llm_explanation_text,
        },
        generated_at=datetime.now(timezone.utc),
    )


def _build_speed_calculator(payload: VpnToolRequest) -> VpnToolResponse:
    source = payload.input
    base_speed = max(1.0, _as_float(source.get("base_speed_mbps"), 100.0))
    distance_km = max(1.0, _as_float(source.get("distance_km"), 1200.0))
    protocol = _norm_text(source.get("protocol")).lower() or "wireguard"
    network_type = _norm_text(source.get("network_type")).lower() or "wifi"
    server_load = _norm_text(source.get("server_load")).lower() or "medium"

    protocol_factor = {
        "wireguard": 0.86,
        "ikev2": 0.8,
        "openvpn-udp": 0.72,
        "openvpn-tcp": 0.62,
    }.get(protocol, 0.72)
    network_factor = {"fiber": 1.0, "wifi": 0.88, "mobile": 0.7}.get(network_type, 0.88)
    load_factor = {"low": 1.0, "medium": 0.86, "high": 0.7}.get(server_load, 0.86)
    distance_factor = max(0.55, 1.0 - (distance_km / 14000.0))

    expected = base_speed * protocol_factor * network_factor * load_factor * distance_factor
    low = max(1.0, expected * 0.82)
    high = max(low, expected * 1.12)
    retention = min(1.0, expected / base_speed)
    score = int(round(retention * 100))
    verdict = _verdict_from_score(score)

    cards = [
        VpnToolCard(title="Expected VPN speed", status="info", value=f"{low:.0f}-{high:.0f} Mbps", detail=f"Base {base_speed:.0f} Mbps."),
        VpnToolCard(title="Protocol factor", status="info", value=protocol, detail=f"Efficiency multiplier {protocol_factor:.2f}."),
        VpnToolCard(title="Distance impact", status="warn" if distance_km > 3000 else "info", value=f"{distance_km:.0f} km", detail=f"Distance multiplier {distance_factor:.2f}."),
        VpnToolCard(title="Server load", status="warn" if server_load == "high" else "pass", value=server_load.title()),
    ]
    actions = [
        "Switch to WireGuard or IKEv2 for better throughput.",
        "Choose a geographically closer server when possible.",
        "Retest on wired/fiber connection for stable benchmark.",
    ]

    return VpnToolResponse(
        tool="vpn-speed-expectation-calculator",
        headline="Estimated VPN throughput generated.",
        verdict=verdict,
        score=score,
        cards=cards,
        actions=actions,
        details={"expected_mbps": round(expected, 2), "retention_ratio": round(retention, 2)},
        generated_at=datetime.now(timezone.utc),
    )


def _build_block_detection(payload: VpnToolRequest) -> VpnToolResponse:
    source = payload.input
    country = _norm_text(source.get("country"))
    cannot_connect = _as_bool(source.get("cannot_connect"), default=False)
    handshake_timeout = _as_bool(source.get("handshake_timeout"), default=False)
    vpn_only_slow = _as_bool(source.get("vpn_only_slow"), default=False)
    port_443_blocked = _as_bool(source.get("port_443_blocked"), default=False)
    works_without_vpn = _as_bool(source.get("works_without_vpn"), default=True)

    score = 82
    causes: list[str] = []
    cards: list[VpnToolCard] = []

    if cannot_connect and works_without_vpn:
        causes.append("VPN transport may be blocked by network or ISP.")
        score -= 24
    if handshake_timeout:
        causes.append("Handshake timeout suggests DPI or aggressive filtering.")
        score -= 18
    if vpn_only_slow:
        causes.append("VPN-specific slowdown may indicate throttling.")
        score -= 15
    if port_443_blocked:
        causes.append("Port 443 blocking detected or suspected.")
        score -= 22

    cards.append(VpnToolCard(title="VPN connection status", status="fail" if cannot_connect else "pass", value="Blocked/unstable" if cannot_connect else "Connected"))
    cards.append(VpnToolCard(title="Handshake behavior", status="warn" if handshake_timeout else "pass", value="Timeout" if handshake_timeout else "Normal"))
    cards.append(VpnToolCard(title="Performance pattern", status="warn" if vpn_only_slow else "pass", value="VPN-only slowdown" if vpn_only_slow else "No clear throttling"))
    cards.append(VpnToolCard(title="Port 443 reachability", status="fail" if port_443_blocked else "pass", value="Blocked/Suspected" if port_443_blocked else "Reachable"))

    score = max(0, min(100, score))
    verdict = _verdict_from_score(score)
    actions = [
        "Switch to TCP 443 and enable obfuscated mode.",
        "Try a different network (mobile hotspot) for A/B comparison.",
        "Use stealth protocol profile and retest handshake.",
    ]
    if country:
        actions.append(f"Check current VPN policy updates for {country}.")

    return VpnToolResponse(
        tool="vpn-block-detection-tool",
        headline="Potential VPN blocking signals analyzed.",
        verdict=verdict,
        score=score,
        cards=cards,
        actions=actions,
        details={"suspected_causes": causes, "country": country},
        generated_at=datetime.now(timezone.utc),
    )


def run_vpn_tool(tool_slug: str, payload: VpnToolRequest) -> VpnToolResponse:
    if tool_slug not in VPN_TOOL_SLUGS:
        raise ValueError("Unsupported VPN tool.")

    response: VpnToolResponse
    if tool_slug == "is-my-vpn-working":
        response = _build_is_my_vpn_working(payload)
    elif tool_slug == "ip-dns-webrtc-leak-tester":
        response = _build_leak_tester(payload)
    elif tool_slug == "vpn-country-compatibility-checker":
        response = _build_country_checker(payload)
    elif tool_slug == "find-best-vpn-for-me-quiz":
        response = _build_best_vpn_quiz(payload)
    elif tool_slug == "vpn-speed-expectation-calculator":
        response = _build_speed_calculator(payload)
    else:
        response = _build_block_detection(payload)

    generation_mode = _safe_str(response.details.get("generation_mode"), max_len=24) or "heuristic"
    generation_scope = _safe_str(response.details.get("generation_scope"), max_len=40) or "heuristic"
    response.details["generation_mode"] = generation_mode
    response.details["generation_scope"] = generation_scope

    if generation_scope == "full-analysis":
        return response

    llm_payload = json_completion(
        system_prompt=(
            "You are a VPN support explainer. "
            "Keep deterministic score/verdict untouched and improve only user-facing explanations. "
            "Return strict JSON."
        ),
        user_prompt=(
            f"Language: {_locale_language_name(payload.locale)}.\n"
            f"Tool: {response.tool}\n"
            "Given this deterministic analysis, rewrite concise user-friendly guidance.\n"
            "Do not change score, verdict, or card titles.\n"
            "Return JSON schema:\n"
            "{"
            "\"headline\":\"...\","
            "\"actions\":[\"...\"],"
            "\"card_details\":{\"Card title\":\"Improved detail text\"},"
            "\"llm_explanation\":\"...\""
            "}\n\n"
            f"Input data: {payload.input}\n"
            f"Headline: {response.headline}\n"
            f"Verdict: {response.verdict}\n"
            f"Score: {response.score}\n"
            f"Cards: {[{'title': c.title, 'status': c.status, 'value': c.value, 'detail': c.detail} for c in response.cards]}\n"
            f"Actions: {response.actions}\n"
        ),
        temperature=0.2,
        max_output_tokens=650,
    )
    if llm_payload:
        llm_headline = _safe_str(llm_payload.get("headline"), max_len=180)
        llm_actions = _safe_str_list(llm_payload.get("actions"), max_items=5, max_len=180)
        llm_explanation = _safe_str(llm_payload.get("llm_explanation"), max_len=450)
        card_details_raw = llm_payload.get("card_details")
        if llm_headline:
            response.headline = llm_headline
        if llm_actions:
            response.actions = llm_actions
        if isinstance(card_details_raw, dict):
            for card in response.cards:
                improved = _safe_str(card_details_raw.get(card.title), max_len=220)
                if improved:
                    card.detail = improved
        if llm_explanation:
            response.details["llm_explanation"] = llm_explanation
        response.details["generation_mode"] = "llm"
        response.details["generation_scope"] = "rewrite-only"

    return response
