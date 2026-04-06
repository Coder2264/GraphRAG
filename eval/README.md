# GraphRAG Evaluation Pipeline

End-to-end benchmark proving that **ToG (Think-on-Graph)** outperforms standard **RAG** and **no-RAG** baselines on multi-hop reasoning questions.

---

## How it works

Documents are designed so facts needed for multi-hop answers are **deliberately scattered** across different sections. Standard RAG top-K retrieval cannot gather all needed chunks in one pass. ToG's graph traversal can.

```
Manual generation (Claude / ChatGPT / Gemini web UIs)
        ↓
eval/import_manual.py      → PDFs + QA JSONs + manifest
        ↓
eval/validate_qa.py        → 4-dimension quality scoring (needs GEMINI_API_KEY)
        ↓
uvicorn main:app --reload  → start GraphRAG server
        ↓
eval/eval_rag.py           → upload docs, query tog/rag/none, LLM judge, report
```

---

## Prerequisites

```bash
# Install reportlab (PDF generation, not part of main app)
pip install "reportlab>=4.0.0"

# GEMINI_API_KEY must be set in .env (used by validate_qa.py and eval_rag.py)
# The main project .env is loaded automatically
```

---

## Step 1 — Generate 100 Documents Manually

Use the prompts below in Claude Pro, ChatGPT Go, and Gemini Pro.
Save each response JSON as `eval/manual_input/doc_NNN.json`.

### Domain split

| Index range | Tool | Domain |
|---|---|---|
| 001–010 | **Claude Pro** | corporate_history |
| 011–020 | **Claude Pro** | scientific_research |
| 021–030 | **Claude Pro** | academic_collaboration |
| 031–040 | **ChatGPT Go** | medical_research |
| 041–050 | **ChatGPT Go** | tech_genealogy |
| 051–060 | **ChatGPT Go** | legal_proceedings |
| 061–070 | **Gemini Pro** | historical_politics |
| 071–080 | **Gemini Pro** | supply_chain |
| 081–090 | **Gemini Pro** | nonprofit_foundations |
| 091–100 | **Gemini Pro** | cultural_institutions |

---

### Claude.ai (Pro) — Setup

**Create a Project** at claude.ai → Projects → New Project.
Paste this as the **Project Instructions** (set once, applies to every conversation in the project):

```
You are a synthetic document generator for a GraphRAG multi-hop reasoning benchmark.
Your job is to create realistic fictional documents where key facts are DELIBERATELY
SCATTERED across different sections so that simple keyword or vector-similarity
retrieval cannot answer multi-hop questions.

OUTPUT FORMAT: Always respond with a single valid JSON object only.
No prose, no markdown fences, no explanation outside the JSON.

CRITICAL ENTITY SCATTERING RULE:
- If entity A connects to entity B, put that fact in one section.
- If entity B connects to entity C, put THAT fact in a DIFFERENT section (at least 2 away).
- Never place two consecutive links of the same answer chain in the same section.
- This is the entire point: defeating top-K chunk retrieval.
```

**Per-document prompt** — paste once per document, fill in DOMAIN / TOPIC / DOCUMENT INDEX:

```
Generate a synthetic document for GraphRAG benchmark testing.

DOMAIN: corporate_history
TOPIC: Founding of fictional biotech NeuroGenix and its acquisition chain
DOCUMENT INDEX: 001

Requirements:
1. Exactly 5 sections, each 450-550 words, labeled "## Section N: [Descriptive Title]"
2. 10-12 named entities total (mix: People, Organizations, Locations, Products, Awards)
3. CRITICAL — scatter connected facts across NON-ADJACENT sections:
   - Section 1: Introduce Company X and its founder Person A
                (do NOT mention where Person A studied)
   - Section 2: Describe Person A's early career at University Y
                (do NOT mention who else trained there)
   - Section 3: Introduce Person B (lead specialist), note their collaboration with Professor C
                (do NOT mention Professor C's institution)
   - Section 4: Describe Professor C's lab at Institute Z
                (do NOT mention who funds Institute Z)
   - Section 5: Name Institute Z's primary funder Organization W and its director Person D
   Adapt this pattern to your domain — the STRUCTURE (one hop per section) is mandatory.
4. Include 1-2 plausible distractor sentences near each entity name that are
   semantically similar to the answer but factually wrong — to confuse keyword search.
5. All names, organizations, and events must be entirely fictional.

After the document, generate exactly 6 multi-hop QA pairs.
Each pair MUST require information from at least 2 different sections.
At least 2 pairs should require 3+ hops.

RESPOND WITH THIS EXACT JSON STRUCTURE (no other text):
{
  "doc_index": "001",
  "domain": "corporate_history",
  "topic": "exact topic string",
  "document_text": "## Section 1: [Title]\n\n[450-550 words]\n\n## Section 2: [Title]\n\n[450-550 words]\n\n## Section 3: [Title]\n\n[450-550 words]\n\n## Section 4: [Title]\n\n[450-550 words]\n\n## Section 5: [Title]\n\n[450-550 words]",
  "entities": [
    {"name": "ExampleCorp", "type": "Organization", "introduced_in_section": 1, "connected_to": ["Alice Marsh"]}
  ],
  "qa_pairs": [
    {
      "id": "doc_001_q1",
      "question": "Which organization funds the lab of the scientist who co-authored with ExampleCorp's lead researcher?",
      "answer": "Organization W",
      "hop_count": 4,
      "entity_chain": ["ExampleCorp", "Person B", "Professor C", "Institute Z", "Organization W"],
      "sections_involved": [1, 3, 4, 5],
      "answer_justification": "ExampleCorp's lead researcher is Person B (section 3); Person B co-authored with Professor C (section 3); Professor C works at Institute Z (section 4); Institute Z is funded by Organization W (section 5)"
    }
  ]
}
```

**Workflow in Claude.ai:**
1. Send the prompt (each time with a new TOPIC from the topic list below)
2. Claude outputs JSON in an Artifact panel
3. Click **Copy** on the Artifact
4. Paste into `eval/manual_input/doc_001.json`
5. Repeat — generate 3-4 docs per conversation, then start a new one

---

### ChatGPT (Go) — Setup

**Start a new conversation.** Paste this as the first message:

```
You are a synthetic document generator for a GraphRAG multi-hop reasoning benchmark.
CRITICAL RULE: Facts forming multi-hop answer chains MUST be in DIFFERENT sections —
never co-located. This defeats simple keyword/vector retrieval and proves multi-hop
reasoning is required.
Output: valid JSON only, no prose outside the JSON object.
```

**Then send per-document prompts** (same format as Claude above, adjusting DOMAIN and TOPIC).

**Workflow in ChatGPT:**
- Each conversation: ~5 documents before context grows too large
- Start a fresh conversation for each batch of 5
- Copy the JSON block, save to `eval/manual_input/doc_031.json`

---

### Gemini Pro — Setup

**Use Gemini Advanced** at gemini.google.com.

Paste this at the start of each session:

```
You are generating synthetic multi-hop test documents for a GraphRAG research benchmark.
KEY REQUIREMENT: Each document must deliberately scatter facts across sections so
that no single 512-character text chunk contains all information needed to answer
the question. Answer chains must cross section boundaries.
Output format: JSON only.
```

**Then send per-document prompts** (same format as Claude).

**Tip:** Use "Export to Docs" on the Gemini response to save it to Google Docs, then copy the JSON portion to `eval/manual_input/doc_061.json`.

---

### Pre-filled Topic List

Use these exact TOPIC + DOMAIN values when sending prompts. All entities and events should be fictional.

| Index | Domain | Topic |
|---|---|---|
| 001 | corporate_history | Founding of fictional biotech NeuroGenix and its acquisition chain |
| 002 | corporate_history | Fictional hedge fund Axiom Capital and its subsidiary network |
| 003 | corporate_history | Rise of fictional aerospace firm Helion Dynamics through mergers |
| 004 | corporate_history | Fictional retail chain Meridian Group's international expansion |
| 005 | corporate_history | Fictional semiconductor company Vantec's patent licensing empire |
| 006 | corporate_history | Fictional shipping conglomerate Tidal Works and its port holdings |
| 007 | corporate_history | Fictional pharmaceutical company Brentex and clinical partner network |
| 008 | corporate_history | Fictional real estate developer Crestholm Partners' project lineage |
| 009 | corporate_history | Fictional energy company Solveig Power and its regulatory battles |
| 010 | corporate_history | Fictional media group Paragon Broadcasting's talent pipeline |
| 011 | scientific_research | Discovery of fictional protein "Auranase" and the labs involved |
| 012 | scientific_research | Development of fictional quantum material "Velkium" |
| 013 | scientific_research | Fictional climate model "ORCA-7" and its contributing institutions |
| 014 | scientific_research | Fictional seismic detection algorithm "QuakeNet" origins |
| 015 | scientific_research | Discovery of fictional species "Helix vorrisi" in remote ecosystem |
| 016 | scientific_research | Development of fictional battery chemistry "FerroCell" |
| 017 | scientific_research | Fictional dark matter detector project "ShadowPulse" |
| 018 | scientific_research | Fictional antibiotic compound "Cetravex" discovery chain |
| 019 | scientific_research | Fictional neuroimaging technique "CortexMap" development |
| 020 | scientific_research | Fictional materials lab collaboration on "aerogel-X" |
| 021 | academic_collaboration | Fictional joint research program "Project Meridian" across 3 universities |
| 022 | academic_collaboration | Fictional cross-institution linguistics study "LexiCore" |
| 023 | academic_collaboration | Fictional ethics board "SAGE" and its member institutions |
| 024 | academic_collaboration | Fictional interdisciplinary lab "Nexus Institute" founding |
| 025 | academic_collaboration | Fictional international fellowship "Harker Prize" network |
| 026 | academic_collaboration | Fictional archaeology consortium "Terra Collective" excavation |
| 027 | academic_collaboration | Fictional AI safety research group "Alignment Bridge" formation |
| 028 | academic_collaboration | Fictional oceanography partnership "DeepScan Alliance" |
| 029 | academic_collaboration | Fictional graduate exchange program "Voss Scholars" founding |
| 030 | academic_collaboration | Fictional joint journal "Synthesis Review" editorial network |
| 031 | medical_research | Clinical trials for fictional drug "Velorazine" for rare neuropathy |
| 032 | medical_research | Fictional gene therapy "VectoShield" for hereditary blindness |
| 033 | medical_research | Fictional oncology trial "CIPHER-3" and its principal investigators |
| 034 | medical_research | Fictional vaccine "ImmuPath-B" regulatory approval chain |
| 035 | medical_research | Fictional cardiology device "PulseGrid" development and testing |
| 036 | medical_research | Fictional rare disease registry "OmegaTrack" and its hospitals |
| 037 | medical_research | Fictional surgical robot "Apex-7" and its academic spinoff |
| 038 | medical_research | Fictional psychiatric study "MindBridge" and its funding trail |
| 039 | medical_research | Fictional biomarker "Trexin" discovery and commercialization |
| 040 | medical_research | Fictional pediatric cancer drug "Curenol" trials and partnerships |
| 041 | tech_genealogy | Fictional AI startup "Latent Labs" and its acquisition by Big Tech |
| 042 | tech_genealogy | Fictional database company "Helix DB" and its open-source spinoffs |
| 043 | tech_genealogy | Fictional semiconductor IP firm "CorePath" patent ownership chain |
| 044 | tech_genealogy | Fictional cloud provider "NimbusStack" and its founding team dispersal |
| 045 | tech_genealogy | Fictional cybersecurity firm "IronShield" and its government contracts |
| 046 | tech_genealogy | Fictional robotics startup "Terrain AI" and its university roots |
| 047 | tech_genealogy | Fictional browser engine "Prism" and its stewardship history |
| 048 | tech_genealogy | Fictional fintech "ClearLedger" and its banking partner network |
| 049 | tech_genealogy | Fictional AR platform "Veil OS" and its hardware spin-offs |
| 050 | tech_genealogy | Fictional IoT standard "MeshLink" and its consortium members |
| 051 | legal_proceedings | Fictional antitrust case involving conglomerate "Orion Holdings" |
| 052 | legal_proceedings | Fictional patent dispute over fictional compound "Helix-9" |
| 053 | legal_proceedings | Fictional class action against fictional insurer "Pinnacle Life" |
| 054 | legal_proceedings | Fictional environmental lawsuit against fictional plant "CrestPower" |
| 055 | legal_proceedings | Fictional securities fraud case involving fictional fund "Apex Ventures" |
| 056 | legal_proceedings | Fictional inheritance dispute over fictional estate "Thornwall Manor" |
| 057 | legal_proceedings | Fictional breach of contract case between fictional tech firms |
| 058 | legal_proceedings | Fictional trademark case over brand "Solaris" across two companies |
| 059 | legal_proceedings | Fictional whistleblower case at fictional pharma "Veron Labs" |
| 060 | legal_proceedings | Fictional international arbitration involving fictional port "Nova Wharf" |
| 061 | historical_politics | Fictional 1987 Morten Accord between nations Valdoria, Kestra, Nurn |
| 062 | historical_politics | Fictional 1962 coup in nation Arvanthia and its aftermath |
| 063 | historical_politics | Fictional cold-war spy exchange at "Bridge Verano" in 1973 |
| 064 | historical_politics | Fictional 1995 peace treaty ending the "Korath Conflict" |
| 065 | historical_politics | Fictional election scandal in nation Pelora in 2003 |
| 066 | historical_politics | Fictional colonial independence movement in nation Serrath, 1958 |
| 067 | historical_politics | Fictional UN resolution 4421 and its drafters |
| 068 | historical_politics | Fictional economic union "Trident Pact" formation in 1981 |
| 069 | historical_politics | Fictional constitutional crisis in Valdoria, 1991 |
| 070 | historical_politics | Fictional refugee crisis arbitration, "Accord of Merrin", 2001 |
| 071 | supply_chain | Fictional lithium battery supply chain from mine "Cerro Azul" to EV maker |
| 072 | supply_chain | Fictional rare earth supply chain for fictional magnet maker "PoleStar" |
| 073 | supply_chain | Fictional textile supply chain from fictional fiber co "NatWeave" to brand |
| 074 | supply_chain | Fictional pharmaceutical API supply chain for drug "Vexorin" |
| 075 | supply_chain | Fictional semiconductor fab supply chain for chip "Apex-X1" |
| 076 | supply_chain | Fictional coffee supply chain from fictional cooperative "Alta Verde" |
| 077 | supply_chain | Fictional automotive parts chain for fictional OEM "Dura Motors" |
| 078 | supply_chain | Fictional packaging supply chain for fictional food brand "Harvest Co" |
| 079 | supply_chain | Fictional aerospace parts chain for fictional jet "Condor 400" |
| 080 | supply_chain | Fictional cold-chain logistics for fictional vaccine "ImmunoPro" |
| 081 | nonprofit_foundations | Fictional foundation "Luminary Trust" and its grantee network |
| 082 | nonprofit_foundations | Fictional environmental NGO "EarthKeep" and its corporate donors |
| 083 | nonprofit_foundations | Fictional education fund "Archer Endowment" and its universities |
| 084 | nonprofit_foundations | Fictional humanitarian org "ShelterFirst" and its field partners |
| 085 | nonprofit_foundations | Fictional arts grant body "Prism Foundation" and its awardees |
| 086 | nonprofit_foundations | Fictional science prize "Voss Medal" and its past recipients |
| 087 | nonprofit_foundations | Fictional animal welfare org "Fauna Watch" and its campaigns |
| 088 | nonprofit_foundations | Fictional microfinance network "TrustBridge" and its sponsors |
| 089 | nonprofit_foundations | Fictional disaster relief fund "Phoenix Relief" and its logistics |
| 090 | nonprofit_foundations | Fictional legal aid org "AccessLaw" and its pro bono firms |
| 091 | cultural_institutions | Fictional art movement "Neo-Brutalism" and its gallery network |
| 092 | cultural_institutions | Fictional film festival "Aurelius Film Week" and its founding patrons |
| 093 | cultural_institutions | Fictional orchestra "Vantara Philharmonic" and its conductor lineage |
| 094 | cultural_institutions | Fictional literary prize "Morrow Prize" and its judging institutions |
| 095 | cultural_institutions | Fictional museum "Holt Institute" and its acquisition history |
| 096 | cultural_institutions | Fictional dance company "Tessera" and its touring partners |
| 097 | cultural_institutions | Fictional cultural exchange program "Meridian Arts" and its embassies |
| 098 | cultural_institutions | Fictional independent publisher "Canopy Press" and its award network |
| 099 | cultural_institutions | Fictional architecture firm "Form & Field" and its civic commissions |
| 100 | cultural_institutions | Fictional photography collective "Aperture Nine" and its gallery chain |

---

## Step 2 — Import + Create PDFs

```bash
python eval/import_manual.py
```

Scans `eval/manual_input/*.json`, validates schema, creates PDFs, and writes `eval/generated/manifest.json`.

Expected output:
```
Imported:    100 documents
QA pairs:    ~600 total
Failed:      0
```

---

## Step 3 — Validate QA Pairs

```bash
python eval/validate_qa.py
```

Uses Gemini (`gemini-2.5-flash-preview-05-20`) to score each QA pair on 4 dimensions:

| Dimension | What it checks |
|---|---|
| **consistency** | Is the answer actually correct per the document? |
| **multi_hop_necessity** | Would a single 512-char chunk be insufficient? |
| **specificity** | Is the question unambiguous with one correct answer? |
| **difficulty** | Would top-5 RAG retrieval fail to retrieve all needed facts? |

Thresholds: mean ≥ 0.7 AND all dimensions ≥ 0.4 → **PASS**

Output: `eval/generated/qa_pairs_validated.json`

---

## Step 4 — Start the GraphRAG Server

```bash
uvicorn main:app --reload
```

Make sure `.env` is configured with all required providers (PostgreSQL, Neo4j, Ollama/Gemini).

---

## Step 5 — Run the Evaluation

```bash
python eval/eval_rag.py
```

For each validated QA pair (pass + warn):
1. Uploads the PDF to the running server
2. Queries `tog`, `rag`, and `none` modes concurrently
3. LLM judge (Gemini Flash) scores retrieval quality and answer quality (0-10)

Output:
- `eval/results/eval_results_{timestamp}.json` — full results
- `eval/results/report_{timestamp}.md` — comparison report

### CLI options

```bash
# Use different modes
python eval/eval_rag.py --modes tog rag none

# Change top-K
python eval/eval_rag.py --top-k 10

# Different server URL
python eval/eval_rag.py --api http://localhost:9000

# Use only a subset of validated QA
python eval/eval_rag.py --validated eval/generated/qa_pairs_validated.json
```

---

## Expected Results

On well-designed multi-hop documents, expect approximately:

| Mode | Retrieval Score | Answer Score |
|---|---|---|
| **tog** | 7-9/10 | 7-8/10 |
| **rag** | 3-5/10 | 3-5/10 |
| **none** | 0/10 | 1-3/10 |

ToG's advantage grows with hop count: expect +3-4 points on 3-hop vs. RAG.

---

## File Structure

```
eval/
├── config.py           # All tunable constants
├── prompts.py          # All prompt strings (generation, validation, judge)
├── utils.py            # PDF creation, Gemini wrapper, JSON parser
├── import_manual.py    # Import JSON files → PDFs + manifest
├── validate_qa.py      # 4-dimension QA quality scoring
├── eval_rag.py         # tog vs rag vs none benchmark
├── README.md           # This file
├── manual_input/       # Drop doc_NNN.json files here (git-ignored)
├── generated/          # Created by import_manual.py (git-ignored)
│   ├── docs/           # PDF files
│   ├── qa_pairs/       # Per-document QA JSONs
│   ├── manifest.json
│   └── qa_pairs_validated.json
└── results/            # Created by eval_rag.py (git-ignored)
    ├── eval_results_*.json
    └── report_*.md
```

---

## Input JSON Schema

Each `eval/manual_input/doc_NNN.json` must follow this schema:

```json
{
  "doc_index": "001",
  "domain": "corporate_history",
  "topic": "Exact topic string",
  "document_text": "## Section 1: Title\n\n[450-550 words]\n\n## Section 2: ...",
  "entities": [
    {
      "name": "NeuroGenix",
      "type": "Organization",
      "introduced_in_section": 1,
      "connected_to": ["Marcus Webb"]
    }
  ],
  "qa_pairs": [
    {
      "id": "doc_001_q1",
      "question": "Who directs the organization that funds the lab of NeuroGenix's lead scientist's mentor?",
      "answer": "Person D",
      "hop_count": 4,
      "entity_chain": ["NeuroGenix", "Person B", "Professor C", "Institute Z", "Person D"],
      "sections_involved": [1, 3, 4, 5],
      "answer_justification": "Lead scientist is Person B (sec 3); mentor is Professor C (sec 3); C's lab is at Institute Z (sec 4); Institute Z director is Person D (sec 5)"
    }
  ]
}
```

Valid domains: `corporate_history`, `scientific_research`, `academic_collaboration`,
`medical_research`, `tech_genealogy`, `legal_proceedings`, `historical_politics`,
`supply_chain`, `nonprofit_foundations`, `cultural_institutions`
