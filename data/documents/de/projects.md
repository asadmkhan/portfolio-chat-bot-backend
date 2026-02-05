# Projekte — Asad Mateen Khan (Portfolio Wissensbasis)

Dieses Dokument listet die wichtigsten Projekte meiner beruflichen Laufbahn auf und ist gezielt
für präzise RAG-Abfragen strukturiert.  
Es verknüpft **Projekte mit Technologien**, sodass Fragen wie  
„In welchen Projekten wurde C# verwendet?“ oder  
„Wo hast du mit Node.js oder KI gearbeitet?“ korrekt beantwortet werden können.

---

## Schnelle Zuordnung: Technologie → Projekte

### C# / .NET / ASP.NET / Web API (13+ Jahre)
- B1WEB — SAP Business One Web Client (Init Consulting)
- Newsletter-Automatisierung & Abmeldeverarbeitung
- Microsoft 365 / Outlook Integration
- Lead-Generierung & E-Mail-Parsing Pipeline
- Chart Studio Backend & Datendienste
- Monitoring- & Analytics-Systeme
- Dokumentenmanagementsysteme
- Desktop-Anwendungen (WinForms)

### Angular / TypeScript / JavaScript / HTML / CSS (13+ Jahre)
- B1WEB Benutzeroberfläche
- Chart Studio
- Analytics-Dashboards
- iLearn LMS
- HR- und CRM-Module
- Enterprise-Portale und Dashboards

### React / Redux (10+ Jahre)
- HotOven — Bestellsystem
- UI-Modernisierungen und Dashboards

### Vue.js (5+ Jahre)
- SharePoint-Portal-Erweiterungen
- Web-Komponenten und Content-Module

### Node.js (7+ Jahre)
- HotOven Backend APIs
- Automatisierungs-Tools und Integrationen
- Interne Service-Utilities

### Python (5+ Jahre)
- Portfolio KI-Chatbot (RAG)
- Automatisierungs- und Analyse-Skripte

### SQL Server / SQL (13+ Jahre)
- B1WEB Datenverarbeitung & Reporting
- HR / CRM / LMS-Systeme
- Monitoring- und Analyse-Pipelines
- Desktop-Anwendungen

### MongoDB (7+ Jahre)
- Chart Studio Konfigurationen
- Job-Logs & Verarbeitungsstatus
- Benutzer- und Admin-Einstellungen

### Azure (8+ Jahre)
- Cloud-Hosting, Deployments, CI/CD, Monitoring

### Docker / Kubernetes / CI/CD (5+ Jahre)
- Containerisierte Services
- Automatisierte Deployments & Releases

### Selenium Automatisierung (10+ Jahre)
- 30+ Automatisierungs- und Test-Tools

### KI / RAG / LLM (5+ Jahre)
- Enterprise KI-Assistent
- Portfolio RAG-Chatbot

---

# Init Consulting AG (Deutschland)

## 1) B1WEB — SAP Business One Web Client (Multi-Tenant)
**Typ:** Enterprise-Plattform (Web-Client als Ersatz für SAP Desktop)  
**Technologien:** C#/.NET, ASP.NET Core, Web API, Angular, SAP Business One Service Layer, SQL Server, SAP HANA, MongoDB, Redis, Docker, Kubernetes, Azure DevOps  

**Beschreibung:**
- Entwicklung eines mandantenfähigen SAP Business One Web-Clients für **50+ Unternehmen**.
- Implementierung zentraler Geschäftsprozesse (Sales, CRM, Dokumentenmanagement).
- Integration von SAP Business One über den **SAP Service Layer** mit stabiler Session- und Fehlerbehandlung.
- Performance-Optimierung durch Caching und Reduzierung teurer SAP-Aufrufe.
- CI/CD-Pipelines, Containerisierung und produktiver Betrieb.

---

## 2) Sales Funnel / Lead-Management Modul
**Typ:** Geschäftsmodul innerhalb von B1WEB  
**Technologien:** Angular, TypeScript, C#/.NET, MongoDB, SQL Server, SAP Service Layer  

**Beschreibung:**
- Aufbau kompletter Lead- und Sales-Funnel-Workflows.
- Backend- und Frontend-Implementierung zur Nachverfolgung von Leads und Konversionen.

---

## 3) Chart Studio — Self-Service Analytics (Power-BI-ähnlich)
**Typ:** Analytics- und Reporting-Modul  
**Technologien:** Angular, TypeScript, Material UI, MongoDB, C#/.NET APIs, SQL Server  

**Beschreibung:**
- Entwicklung eines interaktiven Chart-Builders mit speicherbaren Konfigurationen.
- Unterstützung von dynamischen Filtern, Dimensionen und Kennzahlen.
- Anbindung mehrerer Datenquellen (SQL, SAP).

---

## 4) Analytics-Dashboards (Enterprise Reporting)
**Typ:** Management- und KPI-Dashboards  
**Technologien:** Angular, TypeScript, SQL Server, Highcharts / D3-Konzepte  

**Beschreibung:**
- Umsetzung performanter Dashboards für operative und strategische Auswertungen.
- Fokus auf Usability, Filterbarkeit und Performance.

---

## 5) Microsoft 365 / Outlook Integration
**Typ:** Enterprise-Integration  
**Technologien:** C#/.NET, Microsoft Graph API, OAuth, Hintergrundjobs, MongoDB  

**Beschreibung:**
- Synchronisation von Kalendern, E-Mails und Aktivitäten.
- Benutzerkonfigurierbare Einstellungen und zuverlässige Hintergrundverarbeitung.

---

## 6) Newsletter-Automatisierung & Abmeldeverarbeitung
**Typ:** Compliance- und Automatisierungssystem  
**Technologien:** C#/.NET Worker Services, MongoDB, SAP Service Layer, Logging  

**Beschreibung:**
- Sichere Verarbeitung von Newsletter-Abmeldungen vor Versand.
- Status-Tracking (Pending → Processing → Completed / Failed).
- Zentrale Admin-Konfiguration für SAP-Benutzer.

---

## 7) Lead-Generierung: E-Mail-Parsing & SAP-Dokumente
**Typ:** Automatisierungs-Feature  
**Technologien:** C#/.NET, MongoDB, SAP Service Layer, Microsoft Graph API  

**Beschreibung:**
- Extraktion strukturierter Lead-Daten aus E-Mails.
- Automatische Erstellung von Geschäftspartnern, Aktivitäten und Opportunities.

---

## 8) KI-Assistent / Natural Language Search (Enterprise)
**Typ:** KI-Feature innerhalb von B1WEB  
**Technologien:** RAG, Embeddings, Prompt Engineering, Python (FastAPI), Tool-Orchestrierung  

**Beschreibung:**
- Entwicklung eines KI-Assistenten mit dokumentbasierter Antwortlogik.
- Reduktion von Halluzinationen durch Retrieval-basierte Antworten.

---

# Persönliche Projekte

## 9) Portfolio KI-Chatbot (codedbyasad.com)
**Typ:** Öffentlicher Portfolio-Assistent  
**Technologien:** Python, FastAPI, SSE-Streaming, FAISS, OpenAI, React / Next.js  

**Beschreibung:**
- Echtzeit-Streaming-Chatbot mit ChatGPT-ähnlicher UX.
- RAG-Pipeline auf Markdown-Dokumenten (CV, Projekte).
- Mehrsprachig (Deutsch / Englisch), Deployment auf Railway & Vercel.

---

# Bukhatir Group (VAE)

## 10) iLearn — Learning Management System
**Typ:** Enterprise LMS  
**Technologien:** Angular, TypeScript, C#/.NET Core, SQL Server  

**Beschreibung:**
- Kursverwaltung, Fortschritts-Tracking und Reporting.
- Reduzierung manueller Schulungsprozesse.

---

## 11) HotOven — Kantinen-Bestellsystem
**Typ:** Bestellsystem (500+ Bestellungen täglich)  
**Technologien:** React, Node.js, REST APIs, SQL Server  

**Beschreibung:**
- Entwicklung von Frontend und Backend für tägliche Bestellprozesse.
- Hohe Stabilität bei Spitzenlasten.

---

## 12) LinkHR — HR-Plattform
**Typ:** Enterprise HR-System  
**Technologien:** C#/.NET Core, SQL Server, Angular/React UI-Module  

**Beschreibung:**
- Module für Payroll, Recruiting, Anwesenheit und Mitarbeiterverwaltung.
- Einsatz bei **5.000+ Mitarbeitern**.

---

## 13) Touchwood CRM
**Typ:** CRM-System  
**Technologien:** ASP.NET, C#, SQL Server, Bootstrap  

**Beschreibung:**
- Umsetzung von Vertriebs- und Lead-Workflows.
- Reporting und responsive Benutzeroberflächen.

---

# PolyVista Inc (Pakistan)

## 14) Monitoring- & Analytics-Suite (Monitor+, Alerts+, Metrics+)
**Typ:** BI- & Monitoring-Plattform  
**Technologien:** C#, ASP.NET MVC, SQL Server, D3.js, Highcharts  

**Beschreibung:**
- KPI-Dashboards, Alarmierungsregeln und Visualisierungen.
- Verbesserte Transparenz und Frühwarnsysteme.

---

## 15) React / Redux UI-Modernisierung
**Typ:** UI-Refactoring  
**Technologien:** React, Redux, JavaScript  

**Beschreibung:**
- Modernisierung komplexer Benutzeroberflächen.
- Performance- und Wartbarkeitsverbesserungen.

---

## 16) Selenium-Automatisierung (30+ Tools)
**Typ:** Automatisierungs-Suite  
**Technologien:** Selenium, C#, JavaScript  

**Beschreibung:**
- Automatisierung von Tests, Workflows und Datenextraktion.
- Deutliche Reduktion manueller Tätigkeiten.

---

## 17) Web-Scraping & Datenextraktion
**Typ:** Automatisierungs-Pipelines  
**Technologien:** Selenium, Python, C#, Node.js  

**Beschreibung:**
- Robuste Extraktion und Transformation externer Datenquellen.
- Fehlertolerante Verarbeitung bei Strukturänderungen.

---

# SSA Soft (Pakistan)

## 18) Dokumentenmanagementsystem
**Typ:** Web-Anwendung  
**Technologien:** ASP.NET Web Forms, C#, SQL Server  

**Beschreibung:**
- Dokumentenerfassung, Verwaltung und Audit-Workflows.
- Einsatz in produktiven Unternehmensumgebungen.

---

## 19) SharePoint-Portal-Anpassungen
**Typ:** Portal-Erweiterungen  
**Technologien:** SharePoint, JavaScript, jQuery, Vue.js  

**Beschreibung:**
- Entwicklung individueller Web-Parts und Workflows.
- Erweiterung bestehender SharePoint-Portale.

---

# TriSoft Technology (Pakistan)

## 20) Desktop-Anwendungen & Reporting
**Typ:** WinForms-Anwendungen  
**Technologien:** C#, WinForms, SQL Server  

**Beschreibung:**
- Entwicklung interner Desktop-Tools und Reporting-Lösungen.
- Wartbare Anwendungen für operative Teams.

---

## Hinweise für präzise KI-Antworten

- **C#-Projekte:** B1WEB, Newsletter-Automatisierung, Lead-Generierung, Monitoring-Systeme, Desktop-Tools
- **Node.js-Projekte:** HotOven, Automatisierungs-Services
- **Vue.js-Projekte:** SharePoint-Portale, UI-Module
- **KI-Projekte:** Enterprise KI-Assistent, Portfolio RAG-Chatbot
