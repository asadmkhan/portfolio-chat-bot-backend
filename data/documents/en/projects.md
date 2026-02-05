# Projects — Asad Mateen Khan (Portfolio Knowledge Base)

This document lists key projects across my career and is intentionally structured for accurate RAG retrieval.
It maps projects to skills so questions like “Which project used C#?”, “Where did you use Node.js?”, or
“Which work involved AI?” return precise answers.

---

## Quick Skill → Projects Map (Fast Lookup)

### C# / .NET / ASP.NET / Web API (13+ years)
- B1WEB — SAP Business One Web Client (Init Consulting)
- Newsletter Automation & Unsubscribe Processing (Init Consulting)
- Microsoft 365 / Outlook Integration (Init Consulting)
- Lead Generation & Email Parsing Pipeline (Init Consulting)
- Chart Studio Backend & Data Services (Init Consulting)
- Monitoring & Analytics Suite (PolyVista)
- Document Management System (SSA Soft)
- Desktop Operations Tools (TriSoft)

### Angular / TypeScript / JavaScript / HTML / CSS (13+ years)
- B1WEB UI modules (Init Consulting)
- Chart Studio (Init Consulting)
- Analytics dashboards (Init Consulting)
- iLearn LMS (Bukhatir)
- LinkHR portal modules (Bukhatir)
- Multiple enterprise dashboards and portals (PolyVista)

### React / Redux (10+ years)
- HotOven (Bukhatir)
- React modernization work and UI refactoring (PolyVista)
- Various frontend modules and dashboards (PolyVista)

### Vue.js (5+ years)
- SharePoint portal extensions / web components (earlier projects + enterprise portals)
- Content-driven UI modules (selected projects)

### Node.js (7+ years)
- HotOven backend APIs (Bukhatir)
- Internal automation utilities and integration services (PolyVista)
- Selected API modules and tooling

### Python (5+ years)
- Portfolio RAG Chatbot (Personal)
- Automation scripts, scraping utilities, and AI tooling (selected work)

### SQL Server / SQL (13+ years)
- B1WEB data workflows + reporting
- HR/CRM/LMS applications
- Monitoring dashboards and analytics pipelines
- Desktop systems and reporting tools

### MongoDB (7+ years)
- Chart Studio configurations & saved dashboards
- Job logs / unsubscribe processing logs
- User settings, admin configurations, feature flags

### Azure (8+ years)
- Deployments, CI/CD, App hosting, monitoring, pipelines

### Docker / Kubernetes / CI/CD (5+ years)
- Containerized services and automated deployments (Init Consulting)
- Pipelines and release automation (Azure DevOps)

### Selenium Automation (10+ years)
- 30+ automation tools for workflows, regression testing, and scraping

### AI / RAG / LLM Apps (5+ years)
- Enterprise AI Assistant / Natural Language Search (Init Consulting)
- Portfolio RAG Chatbot (Personal)

---

# Init Consulting AG (Germany)

## 1) B1WEB — SAP Business One Web Client (Multi-tenant)
**Type:** Enterprise platform (Web client replacing SAP desktop workflows)  
**Skills:** C#/.NET, ASP.NET Core, Web API, Angular, SAP B1 Service Layer, SQL Server, SAP HANA, MongoDB, Redis, Docker, Kubernetes, Azure DevOps  
**Highlights:**
- Built and evolved a multi-tenant SAP Business One web client used by **50+ companies**.
- Implemented core business modules and workflows (Sales, CRM, document handling).
- Integrated SAP Business One via **SAP Service Layer**, handling sessions, retries, and reliability patterns.
- Optimized performance via caching (Redis) and reducing expensive SAP round-trips.
- Deployed services via CI/CD pipelines and maintained production reliability through logs and monitoring.

---

## 2) Sales Funnel / Lead Management Module
**Type:** Business module inside B1WEB  
**Skills:** Angular, TypeScript, C#/.NET, MongoDB, SQL Server, SAP Service Layer  
**Highlights:**
- Built lead capture, qualification, and sales funnel workflows.
- Implemented backend services and UI to track lead lifecycle and conversion metrics.

---

## 3) Chart Studio — Power BI-like Analytics Builder
**Type:** Self-service analytics module  
**Skills:** Angular, TypeScript, Material UI, MongoDB, C#/.NET APIs, SQL Server, dynamic filtering patterns  
**Highlights:**
- Built interactive chart builder UI (dimensions/measures) with saved configurations stored in MongoDB.
- Implemented per-chart filters, field-driven filter generation, and chart deletion workflows.
- Supported multiple data sources including SQL and SAP queries.

---

## 4) Analytics Dashboards (Enterprise Reporting)
**Type:** Dashboards and reporting UI  
**Skills:** Angular, TypeScript, SQL Server, visualization tooling (Highcharts/D3 patterns), backend APIs  
**Highlights:**
- Delivered dashboards for KPIs, reporting, and operational visibility.
- Focused on performance, responsiveness, and user-friendly filtering/search.

---

## 5) Microsoft 365 / Outlook Integration (Graph API)
**Type:** Enterprise integration  
**Skills:** C#/.NET, Microsoft Graph API, OAuth, background jobs, user settings, MongoDB  
**Highlights:**
- Built calendar sync, email workflows, activity syncing, and configurable user settings.
- Implemented reliable background processing with retries and status tracking.

---

## 6) Newsletter Automation & Unsubscribe Processing System
**Type:** Compliance + automation pipeline  
**Skills:** C#/.NET Worker Services, Quartz.NET/Hangfire patterns, MongoDB, SAP Service Layer, logging & tracking  
**Highlights:**
- Implemented unsubscribe processing that must run before sending newsletters.
- Stored processing logs and statuses (Pending → Processing → Completed/Failed).
- Built admin configuration for a dedicated SAP user to execute unsubscribe operations safely.

---

## 7) Lead Generation: Email Parsing + SAP Document Creation
**Type:** Automation feature for converting inbound email leads to SAP entities  
**Skills:** C#/.NET, MongoDB, parsing logic, SAP Service Layer, Microsoft Graph API (email retrieval)  
**Highlights:**
- Extracted structured lead data from email content (company, contact, phone, email).
- Designed a flow to create Business Partners, Activities, Opportunities and store references to MongoDB.

---

## 8) AI Assistant / Natural Language Search (Enterprise)
**Type:** AI feature inside B1WEB  
**Skills:** RAG, embeddings, prompt engineering, provider-agnostic design, Python services (FastAPI), tool/function calling patterns  
**Highlights:**
- Designed AI integration architecture with retrieval grounded responses.
- Built AI workflows to answer from internal knowledge while reducing hallucinations.

---

# Personal Projects

## 9) Portfolio RAG Chatbot (codedbyasad.com)
**Type:** Public portfolio AI assistant  
**Skills:** Python, FastAPI, SSE streaming, FAISS, sentence-transformers, OpenAI streaming, React/Next.js, Vercel, Railway  
**Highlights:**
- Built a production-ready chatbot with real-time streaming (ChatGPT-like UX).
- Implemented RAG ingestion: markdown → chunking → embeddings → FAISS.
- Multilingual support: English + German indexes.
- Deployed backend on Railway and integrated widget into Next.js portfolio on Vercel.

---

# Bukhatir Group (UAE)

## 10) iLearn — Learning Management System (LMS)
**Type:** Enterprise LMS  
**Skills:** Angular, TypeScript, C#/.NET Core, SQL Server, API design  
**Highlights:**
- Built learning flows, course authoring, progress tracking, and reporting.
- Reduced manual training coordination by introducing self-service functionality.

---

## 11) HotOven — Canteen Ordering System
**Type:** Ordering system (500+ daily orders, multiple schools)  
**Skills:** React, Node.js, REST APIs, SQL Server, JavaScript/HTML/CSS  
**Highlights:**
- Built React UI and Node.js APIs supporting high-frequency daily ordering.
- Improved order accuracy and reduced wait times using streamlined flows.

---

## 12) LinkHR — HR Platform (5,000+ employees)
**Type:** Enterprise HR system  
**Skills:** C#/.NET Core, SQL Server, frontend modules (Angular/React patterns), REST APIs  
**Highlights:**
- Delivered payroll, recruitment, attendance and employee workflow features.
- Worked across multiple business units with production support responsibility.

---

## 13) Touchwood CRM
**Type:** CRM system  
**Skills:** ASP.NET, C#, SQL Server, frontend UI patterns, Bootstrap  
**Highlights:**
- Built CRM workflows for sales execution and lead tracking.
- Delivered responsive UI and integrated reporting.

---

# PolyVista Inc (Pakistan)

## 14) Monitor+ / Alerts+ / Metrics+ — Monitoring & Analytics Suite
**Type:** BI dashboards + monitoring tools  
**Skills:** C#, .NET, ASP.NET MVC, SQL Server, D3.js, Highcharts, JavaScript  
**Highlights:**
- Built dashboards, alerting rules, and visualizations for executive and operational monitoring.
- Improved proactive incident detection and KPI visibility.

---

## 15) React + Redux Modernization (Enterprise UI)
**Type:** UI modernization and scalability improvements  
**Skills:** React, Redux, JavaScript, backend integration  
**Highlights:**
- Refactored UI flows for performance and maintainability.
- Improved complex UI workflows and ensured stability under load.

---

## 16) Selenium Automation Tooling (30+ tools)
**Type:** Automation suite  
**Skills:** Selenium, C#, JavaScript, workflow automation, regression automation  
**Highlights:**
- Built automation tools for data extraction, workflow testing, and reliability.
- Improved testing coverage and reduced manual repetitive operations.

---

## 17) Web Scraping / Data Extraction Utilities
**Type:** Automation + scraping pipelines  
**Skills:** Selenium, Python, C#, Node.js, data parsing, SQL Server  
**Highlights:**
- Automated extraction from web sources and transformed data into usable formats.
- Built stable pipelines that tolerate website structure changes.

---

## 18) Google Trends Analytics Integration
**Type:** Analytics tool  
**Skills:** C#, .NET, SQL Server, API integrations, Python/Node.js utilities  
**Highlights:**
- Integrated trend signals into analytics dashboards and reporting.
- Enabled exportable insights used for research and monitoring.

---

# SSA Soft (Pakistan)

## 19) Document Management System (ASP.NET Web Forms)
**Type:** Document scanning + management  
**Skills:** ASP.NET Web Forms, C#, SQL Server, workflow modules  
**Highlights:**
- Built document scanning workflows and management UI.
- Delivered features for managing documents and maintaining auditability.

---

## 20) SharePoint Portal Customizations
**Type:** SharePoint portals  
**Skills:** SharePoint, JavaScript, jQuery, Vue.js (selected UI modules), HTML/CSS  
**Highlights:**
- Extended portals with custom workflows and UI modules.
- Built business-specific web parts and portal enhancements.

---

# TriSoft Technology (Pakistan)

## 21) Desktop Tools & Reporting (WinForms)
**Type:** Desktop applications  
**Skills:** C#, WinForms, SQL Server, reporting  
**Highlights:**
- Built desktop utilities for operational tracking and reporting.
- Delivered maintainable WinForms applications for internal teams.

---

## Notes for RAG Accuracy

- If a user asks “projects using C#”, the best matches are: **B1WEB, Newsletter Automation, Lead Generation, Monitoring Suite, Document Management, WinForms tools**.
- If a user asks “projects using Node.js”, the best matches are: **HotOven, automation utilities, integrations**.
- If a user asks “projects using Vue.js”, the best matches are: **SharePoint portals and UI components**.
- If a user asks “projects using AI”, the best matches are: **Enterprise AI Assistant, Portfolio RAG Chatbot**.
