#!/usr/bin/env python3
"""Swedish bank-transaction classifier, abstention-first.

Pipeline: normalize text -> merchant key -> structural rules -> exact lexicon -> corroborated weak lexicon
-> Claude Haiku 4.5 for unresolved unique keys (abstain below LLM_MIN_CONF) -> sign sanity -> asserts.

Usage: python classify.py input.csv [-o predictions.csv] [--train train.csv] [--no-llm] [--no-train-map]
Reads ANTHROPIC_API_KEY from the environment only. Model via CLASSIFIER_MODEL (default Haiku 4.5).
"""
import argparse, csv, io, json, os, re, sys, time, statistics
from collections import Counter, defaultdict

# LLM answers below this confidence are emitted as "unknown" (wrong costs a point, unknown costs nothing)
LLM_MIN_CONF = 0.72
# a positive deposit at least this large is treated as salary only with a second signal (LÖN / employer pattern)
SALARY_MIN = 12000
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
PRICE = {"input": 1.00, "output": 5.00, "cache_write": 1.25, "cache_read": 0.10}   # USD per 1M tokens, Haiku 4.5

CATS = ["income.employment", "income.public", "income.sales", "income.capital",
        "spend.housing", "spend.groceries", "spend.dining", "spend.transport", "spend.shopping",
        "spend.leisure", "spend.health", "spend.family", "spend.gambling",
        "credit.loan", "credit.collections",
        "transfer.own", "transfer.savings", "transfer.person", "transfer.cash", "unknown"]

# ---------------------------------------------------------------- loading
def parse_amount(v):
    """'-1 060,00', '−1060.00', '1060.00', '1,060.00', '1.060,00', '' -> float (0.0 if unparseable)."""
    t = str(v or "").replace("−", "-").replace(" ", "").replace(" ", "").strip()
    if not t: return 0.0
    if "," in t and "." in t:
        dec = "," if t.rfind(",") > t.rfind(".") else "."
        t = t.replace("." if dec == "," else ",", "").replace(dec, ".")
    elif "," in t:
        head, tail = t.rsplit(",", 1)
        t = (head + "." + tail) if len(tail) <= 2 else (head + tail)
    try: return float(t)
    except ValueError: return 0.0

REQUIRED = ["id", "customer", "bank", "account", "account_name", "account_type", "date", "amount_sek", "text"]

def load_csv(path):
    raw = open(path, encoding="utf-8-sig").read().replace(" ", " ").splitlines()
    fixed = []
    for l in raw:
        l = l.rstrip("\r")
        if l.startswith('"') and l.endswith('"') and "," in l[1:-1]:
            inner = l[1:-1]
            if inner.count('"') == 0 or '""' in inner:      # whole-line-quoted export (Google Sheets)
                l = inner.replace('""', '"')
        fixed.append(l)
    rows = list(csv.DictReader(io.StringIO("\n".join(fixed))))
    for n, r in enumerate(rows):
        for c in REQUIRED: r.setdefault(c, "")
        for c in list(r):
            if r[c] is None: r[c] = ""
        if not r["id"]: r["id"] = f"row{n+1}"
        r["text"] = str(r.get("text") or "")
        r["amount"] = parse_amount(r.get("amount_sek") or r.get("amount") or "")
    return rows

# ---------------------------------------------------------------- normalization
PROVIDERS = r"^(KLARNA\*|ZETTLE_\*|ZETTLE\*|SUMUP \*|IZ \*|PAYPAL \*|GOOGLE \*|TRUSTLY\*|SQ \*|APPLE\.COM/BILL)\s*"
SAVINGS_WORDS = ("SPAR", "RÄNTEKONTO", "RANTEKONTO", "BUFFERT", "BARN", "SAVING", "ISK", "KAPITAL", "FOND", "PLACERING")
EVERYDAY_WORDS = ("PERSONKONTO", "PRIVATKONTO", "LÖNEKONTO", "LONEKONTO", "VARDAGSKONTO", "HUSHÅLLSKONTO", "HUSHALLSKONTO",
                  "ALLKONTO", "TRANSAKTIONSKONTO", "BASKONTO", "KREDITKORT", "MASTERCARD", "VISA", "BETALKONTO", "LÖNEKTO")

# Structural phrase synonyms (conservative: each alternative is a specific banking phrase, anchored where possible)
RX_TRANSFER = re.compile(r"^(INTERN\s+)?(ÖVERFÖRING|OVERFORING|ÖVERF|OVERF|EGEN\s+ÖVERF\w*|EGEN\s+OVERF\w*|KONTOÖVERFÖRING|KONTOOVERFORING|(TILL|FRÅN|FRAN)\s+KONTO)\b")
RX_TRANSFER_STRIP = re.compile(r"^(INTERN\s+)?(ÖVERFÖRING|OVERFORING|ÖVERF|OVERF|EGEN\s+ÖVERF\w*|EGEN\s+OVERF\w*|KONTOÖVERFÖRING|KONTOOVERFORING)\s*(FRÅN|FRAN|TILL)?\s*(KONTO)?\s*")
RX_CASH = re.compile(r"\b(UTTAG|ATM|BANKOMAT|KONTANTUTTAG|UTTAGSAUTOMAT|CASH\s+WITHDRAWAL|KONTANT\s+UTTAG)\b")
RX_INTEREST = re.compile(r"^(RÄNTA|RANTA|RÄNTOR|RANTOR|KAPITALISERAD\s+RÄNTA)\b|\bUTDELNING\b")
RX_SALARY = re.compile(r"^(LÖN|LON|SALARY|LÖNEUTBETALNING|LONEUTBETALNING|LÖNEUTBET|NETTOLÖN)\b")
RX_CARD_IN = re.compile(r"^(INBETALNING|INBETALT|BETALNING\s+MOTTAGEN|TACK\s+FÖR\s+DIN\s+BETALNING|TACK\s+FOR\s+DIN\s+BETALNING)")
RX_CARD_IN_ANY = re.compile(r"\b(INBETALNING|INBETALT|BETALNING\s+MOTTAGEN|TACK)\b")
RX_SWISH = re.compile(r"^SWISH\s*(BETALNING|INBETALNING|UTBETALNING|TILL|FRÅN|FRAN|FROM|TO|MOTTAGEN|SKICKAD)?\s*(.*)$")
RX_CARD_BILL = re.compile(r"\b(KORT|KREDITKORT|KORTFAKTURA|KREDITKORTSFAKTURA|KREDITKORTSBETALNING|KORTRÄKNING|MASTERCARD|VISA|AMEX|EUROCARD)\b")

def clean(text):
    t = (text or "").replace(" ", " ").upper().strip()
    t = re.sub(r"\s*/\d{2}-\d{2}-\d{2}$", "", t)                          # SEB date tail
    t = re.sub(r"^(KORTKÖP|KORTKOP|KORT)\s+\d{6}\s*", "", t)                # Nordea card prefix with date
    t = re.sub(r"^(KORTKÖP|KORTKOP)\s+", "", t)                             # card prefix without date
    t = re.sub(r"\s+(EUR|DKK|NOK|USD|GBP|PLN|CHF|THB|TRY)\s+[\d.]+$", "", t)  # fx tail
    t = re.sub(r"\s+", " ", t).strip()
    return t

def parse(r):
    """Return dict(kind, merchant, message, person) from one row."""
    t = clean(r["text"])
    acct_type = r.get("account_type", "")
    out = {"kind": "merchant", "merchant": t, "message": "", "person": ""}
    if not t: out["kind"] = "empty"; return out
    if RX_TRANSFER.match(t):
        out["kind"] = "transfer"
        out["merchant"] = RX_TRANSFER_STRIP.sub("", t)
        return out
    if RX_CASH.search(t):
        out["kind"] = "cash"; return out
    if RX_INTEREST.search(t):
        out["kind"] = "interest"; return out
    if RX_SALARY.match(t):
        out["kind"] = "salary"; return out
    if acct_type == "credit_card" and RX_CARD_IN.match(t):
        out["kind"] = "cardpayment_in"; return out
    m = RX_SWISH.match(t)
    if m:
        rest = m.group(2).strip()
        toks = rest.split()
        if re.search(r"\b(AB|HB|KB|LTD|AS|APS|OY)\b", rest) or lex_match(merchant_key(rest))[1] == "exact":
            out["kind"] = "merchant"; out["merchant"] = rest; return out     # business paid via Swish
        if len(toks) >= 2 and re.fullmatch(r"[A-ZÅÄÖÉ\-]+", toks[0]) and re.fullmatch(r"[A-ZÅÄÖ]", toks[1]):
            out["kind"] = "swish"; out["person"] = " ".join(toks[:2]); out["message"] = " ".join(toks[2:]); return out
        out["kind"] = "swish"; out["person"] = rest; return out
    t2 = re.sub(r"^(AUTOGIRO|AG|E-FAKTURA|EFAKTURA)\s+", "", t)
    t2 = re.sub(r"^(BETALNING BG|BETALNING PG|BETALNING|BG|PG|INSÄTTNING|INSATTNING|INBETALNING FRÅN|INBETALNING)\s+", "", t2)
    t2 = re.sub(r"\s+(BG|PG)$", "", t2)
    t2 = re.sub(PROVIDERS, "", t2).strip()
    out["merchant"] = t2
    return out

def merchant_key(m):
    """Strip store numbers / reference codes for lookup, caching and the LLM."""
    toks = [x for x in m.split() if not re.fullmatch(r"\d+", x) and not (len(x) >= 5 and len(re.findall(r"\d", x)) >= 2)]
    return " ".join(toks) if toks else m

# ---------------------------------------------------------------- lexicon
# EXACT: unambiguous brands / multi-word names / distinctive words. Matched anywhere in the key on word boundaries.
#        A trailing "*" means prefix match (FASTIGHET* hits FASTIGHETER, FASTIGHETS AB).
LEX_EXACT = {
 "spend.groceries": ["ICA MAXI","ICA NARA","ICA NÄRA","ICA SUPERMARKET","ICA KVANTUM","WILLYS","HEMKOP","HEMKÖP","LIDL","CITY GROSS",
                     "COOP KONSUM","COOP FORUM","COOP EXTRA","STORA COOP","MATHEM","MATSMART","PINGO DOCE","REMA 1000","TESCO","SAINSBURY*",
                     "ALDI","CARREFOUR","EUROCASH","LIVSMEDEL*","MATBUTIK*"],
 "spend.dining": ["ESPRESSO HOUSE","WOLT","FOODORA","UBER *EATS","UBER EATS","MCDONALDS","MCDONALD'S","BURGER KING","SUBWAY","PIZZA HUT","SUSHI YAMA",
                  "JOE & THE JUICE","WAYNES COFFEE","VAPIANO","O'LEARYS","OLEARYS","SALLADSBAREN",
                  "STARBUCKS","TACO BAR","SIBYLLA","PRESSBYRAN","PRESSBYRÅN","7-ELEVEN","SEVEN ELEVEN","MAX BURGERS","MAX HAMBURGARE",
                  "DOMINOS","DOMINO'S","KFC","PINCHOS","BASTARD BURGERS","BRÖDERNAS","BRODERNAS","GATEAU","FABRIQUE"],
 "spend.transport": ["SL CENTRALEN","SKÅNETRAFIKEN","SKANETRAFIKEN","VASTTRAFIK","VÄSTTRAFIK","OSTGOTATRAFIKEN","ÖSTGÖTATRAFIKEN","VOI TECHNOLOGY",
                     "TIER MOBILITY","BOLT.EU","UBER *TRIP","UBER TRIP","TAXI KURIR","CABONLINE","CIRCLE K","OKQ8","EASYPARK","PARKSTER","APCOA","AIMO PARK",
                     "ØRESUNDSBRON","ORESUNDSBRON","ÖRESUNDSBRON","CP COMBOIOS","FLYGBUSSARNA","FLIXBUS","SJ AB","ARLANDA EXPRESS","TRAFIKVERKET",
                     "LÄNSTRAFIK*","LANSTRAFIK*","KOLLEKTIVTRAFIK*","PARKERING*","BENSIN*","DRIVMEDEL*"],
 "spend.shopping": ["ZALANDO","H&M","LYKO","AMAZON","IKEA","JYSK","CLAS OHLSON","ELGIGANTEN","NETONNET","KJELL & CO","KJELL&CO","MEDIAMARKT","MEDIA MARKT",
                    "BAUHAUS","BYGGMAX","JULA","HORNBACH","ARKET","ZARA","BOOZT","NELLY","ELLOS","CDON","ADLIBRIS","KICKS","LINDEX","KAPPAHL","ÅHLÉNS","AHLENS",
                    "STADIUM","INTERSPORT","APPLE STORE","WEBHALLEN","RUSTA","ÖOB","DOLLARSTORE","GINA TRICOT","MONKI","WEEKDAY","UNIQLO","BOKUS",
                    "TEKNIKMAGASINET","GRANIT","HEMTEX","LAGERHAUS","PLANTAGEN","BLOMSTERLANDET","BILTEMA","SELLPY","VINTED","TRADERA","BLOCKET",
                    "MIO MÖBLER","CHILLI","ELLOS","CERVERA","INDISKA","ÅHLENS","NORMAL","DESENIO","APOTEA SHOP","CUBUS","DRESSMANN","MQ"],
 "spend.leisure": ["SPOTIFY","NETFLIX","HBO MAX","HBO NORDIC","DISNEY PLUS","DISNEY+","VIAPLAY","STORYTEL","BOOKBEAT","YOUTUBE PREMIUM","YOUTUBE",
                   "STEAMGAMES","STEAM GAMES","EPICGAMES","EPIC GAMES","PLAYSTATION","NINTENDO","XBOX","GOOGLE PLAY","SATS","NORDIC WELLNESS","FRISKIS*",
                   "ACTIC","FITNESS24SEVEN","FITNESS24","SF BIO","FILMSTADEN","TICKETMASTER","LISEBERG","GRÖNA LUND","GRONA LUND","LEGOLAND","LALANDIA",
                   "NORWEGIAN","LUFTHANSA","RYANAIR","BOOKING.COM","HOTELS.COM","AIRBNB","SCANDIC","SKISTAR","TWITCH","DISCORD","KOLMÅRDEN","KOLMARDEN",
                   "SKANSEN","JUNIBACKEN","FURUVIK","NORDIC CHOICE","RADISSON",
                   "APPLE.COM/BILL","APPLE MUSIC","TIDAL","AUDIBLE","KINDLE","BLIZZARD","EA PLAY","UBISOFT","PADEL*"],
 "spend.health": ["APOTEK","APOTEKET","APOTEA","KRONANS APOTEK","FOLKTANDVÅRDEN","FOLKTANDVARDEN","PATIENTAVG*","PRAKTIKERTJÄNST","PRAKTIKERTJANST",
                  "VÅRDCENTRAL*","VARDCENTRAL*","MIN DOKTOR","AKADEMIKLINIKEN","SYNSAM","SPECSAVERS","SYNOPTIK","TANDLÄKAR*","TANDLAKAR*","TANDVÅRD*",
                  "TANDVARD*","DOKTOR.SE","CAPIO","ALERIS","SJUKHUS*","LÄKARHUS*","SMILE TANDVÅRD","DISTRIKTSTANDV*"],
 "spend.housing": ["TELIA","TELENOR","TELE2","COMVIQ","HI3G","HALEBOP","VIMLA","BAHNHOF","BREDBAND*","ELLEVIO","E.ON","EON ","VATTENFALL","FORTUM",
                   "GÖTEBORG ENERGI","GOTEBORG ENERGI","ENERGI*","ELNÄT*","ELNAT*","ELHANDEL*","HEIMSTADEN","AKELIUS","WALLENSTAM","MKB FASTIGHETS",
                   "RIKSHEM","VICTORIA PARK","STENA FASTIGHETER","BALDER","SVENSKA BOSTÄDER","SVENSKA BOSTADER","BOSTADSBOLAGET","FAMILJEBOSTÄDER",
                   "STOCKHOLMSHEM","FASTIGHET*","BOSTAD*","BOSTÄDER","HYRES*","HEMFÖRSÄKRING","FOLKSAM","TRYGG HANSA","TRYGG-HANSA","LÄNSFÖRSÄKRINGAR",
                   "LANSFORSAKRINGAR","IF SKADEF*","HEDVIG","FLYTTHJÄLP*","FLYTTHJALP*","FLYTTFIRMA*","KRAFTRINGEN","JÄMTKRAFT","SKELLEFTEÅ KRAFT",
                   "TELENOR SVERIGE","FIBER*","SAMFÄLLIGHET*","HSB","RIKSBYGGEN","STOCKHOLM EXERGI"],
 "spend.family": ["BARNOMSORG","FÖRSKOLA*","FORSKOLA*","FRITIDSHEM*","SKOLAVGIFT*","BARNOMSORGSAVG*","DAGIS*","INTERNATIONELLA ENGELSKA SKOLAN"],
 "spend.gambling": ["UNIBET","BETSSON","SVENSKA SPEL","ATG.SE","LEOVEGAS","BET365","BETWAY","CASUMO","KINDRED","POKERSTARS","MR GREEN","POSTKODLOTTERIET",
                    "BINGOLOTTO","CASINO","LOTTERI*","MARIA CASINO","COMEON","NORDICBET","PAF","BETHARD","VIDEOSLOTS","SPELA.SE","LOTTO"],
 "credit.loan": ["KLARNA AB","KLARNA BANK","QLIRO","WALLEY","SANTANDER","RESURS","NORDAX","ANYFIN","SBAB","BOLAN","BOLÅN","HYPOTEK","AMORTERING",
                 "COLLECTOR","LENDIFY","LENDO","MARGINALEN","IKANO BANK","SVEA","BANK NORWEGIAN","NORDEA FINANS","SEB KORT","SWEDBANK KORT",
                 "NORDEA KREDITKORT","AMERICAN EXPRESS","AMEX","ENTERCARD","REMEMBER","MOANK","BRIXO","FERRATUM","THORN","AVIDA","NORTHMILL","KREDITKORT"],
 "credit.collections": ["INTRUM","KRONOFOGDEN","ALEKTUM","LOWELL","SERGEL","LINDORFF","GOTHIA","INKASSO*","VISMA COLLECTORS","ALEKTUM GROUP","SVEA INKASSO"],
 "income.public": ["FÖRSÄKRINGSKASSAN","FORSAKRINGSKASSAN","PENSIONSMYNDIGHETEN","ARBETSFÖRMEDLINGEN","ARBETSFORMEDLINGEN","A-KASSA","AKASSA","ALFAKASSAN",
                   "BARNBIDRAG","BOSTADSBIDRAG","FÖRÄLDRAPENNING","SJUKPENNING","STUDIEBIDRAG","AFA FÖRSÄKRING"],
}
# WEAK: short or generic tokens that can name several things. Only trusted when the token LEADS the merchant key
# (brand position, e.g. "MAX GBG", "COOP MAJORNA", "KRY") and the amount sign fits the category. Otherwise -> LLM.
LEX_WEAK = {
 "spend.groceries": ["ICA","COOP","NETTO","TEMPO","HANDLARN","PARADISET","BUDGET","MATÖPPET","MATOPPET","NÄRA","LIVS"],
 "spend.dining": ["MAX","PIZZA","PIZZERIA","SUSHI","RESTAURANG","RESTAURANT","KAFE","CAFE","CAFÉ","FIKA","FOODTRUCK","BISTRO","BAR","THAI","KEBAB","GRILL",
                  "LUNCH","BAGERI","KONDITORI","WAYNES","KROG","PUB","BURGER","BAKERY","COFFEE","KAFFE","GLASS","FOOD"],
 "spend.transport": ["SL","UL","SJ","MTR","VOI","TIER","BOLT","LIME","TAXI","PREEM","INGO","Q8","ST1","SHELL","AIMO","PARKERING","P-HUS","PARKING","TÅG","BUSS"],
 "spend.shopping": ["HM","MIO","XXL","NIKE","ADIDAS","JULA","BUTIK","SHOP","STORE"],
 "spend.leisure": ["HBO","DISNEY","STEAM","SAS","HOTEL","HOTELL","MUSEUM","MUSEU","BIO","CINEMA","BOWLING","GYM","SPA","TEATER","ARENA","ZOO"],
 "spend.health": ["KRY","DOKTOR","OPTIKER","TANDLÄKARE","KLINIK","VÅRD","VARD"],
 "spend.housing": ["TRE","EL","HYRA","BRF","FLYTT","FÖRSÄKRING","FORSAKRING","KRAFT","VATTEN","FJÄRRVÄRME"],
 "spend.family": ["SKOLA","FRITIDS","FÖRSKOLAN"],
 "spend.gambling": ["ATG","BINGO","BET","SPEL"],
 "credit.loan": ["KORT","KREDIT","VISA","MASTERCARD","LÅN","LAN","BANK","FINANS","CSN"],
 "income.public": ["CSN"],
}
SALES = ("TRADERA", "BLOCKET", "VINTED", "SELLPY", "PLICK", "MARKETPLACE")
EMPLOYER_PAT = re.compile(r"\b(LÖN|LON|SALARY|AB|HB|KB|REGION|KOMMUN|STAD|LANDSTING|UNIVERSITET|HÖGSKOLA|HOGSKOLA|MYNDIGHET|FÖRVALTNING|"
                          r"SVERIGE|GROUP|CONCERN|KONCERN|LTD|INC|GMBH|AS|APS|OY|SKOLA|SKOLAN|SJUKHUS)\b")

def _compile(entries, leading):
    pats = []
    for e in entries:
        e = e.upper(); prefix = e.endswith("*"); core = re.escape(e.rstrip("* "))
        tail = r"[A-ZÅÄÖÉ]*" if prefix else ""
        pats.append((len(e), re.compile(("^" if leading else r"(?<![A-ZÅÄÖÉ&'])") + core + tail + r"(?![A-ZÅÄÖÉ])")))
    return pats
_EXACT = {cat: _compile(v, False) for cat, v in LEX_EXACT.items()}
_WEAK = {cat: _compile(v, True) for cat, v in LEX_WEAK.items()}

def lex_match(key):
    """-> (category, 'exact'|'weak'|None). Longest exact entry wins; weak only if nothing exact matched."""
    best = None
    for cat, pats in _EXACT.items():
        for n, p in pats:
            if p.search(key) and (best is None or n > best[0]): best = (n, cat)
    if best: return best[1], "exact"
    for cat, pats in _WEAK.items():
        for n, p in pats:
            if p.search(key) and (best is None or n > best[0]): best = (n, cat)
    return (best[1], "weak") if best else (None, None)

def sign_ok(cat, amt):
    if cat.startswith("income."): return amt > 0
    if cat == "spend.gambling" or cat.startswith("transfer."): return True
    return amt < 0

# Swish message words. Only unambiguous words are mapped (per the brief's category definitions);
# any other message -> unknown. No message -> transfer.person.
MSG_MAP = {"HYRA": "spend.housing", "HYRAN": "spend.housing", "EL": "spend.housing", "ELEN": "spend.housing", "ELRÄKNING": "spend.housing",
           "MATKASSE": "spend.groceries", "MATKASSEN": "spend.groceries",
           "PIZZA": "spend.dining", "LUNCH": "spend.dining", "LUNCHEN": "spend.dining", "MIDDAG": "spend.dining", "MIDDAGEN": "spend.dining",
           "KONSERT": "spend.leisure", "KONSERTEN": "spend.leisure", "BILJETTER": "spend.leisure", "BILJETT": "spend.leisure"}
# deliberately NOT mapped (ambiguous): MAT, BIO, PRESENT, FIKA, RESA, ...

# ---------------------------------------------------------------- structural rules
def find_mirror(r, rows_by_cust):
    """Row in another account of the same customer with the opposite amount on the same day (or ±1 day)."""
    cands = [o for o in rows_by_cust.get(r.get("customer"), [])
             if o is not r and o.get("account") != r.get("account") and abs(o["amount"] + r["amount"]) < 0.005]
    for o in cands:
        if o.get("date") == r.get("date"): return o
    for o in cands:
        try:
            d1, d2 = r.get("date", "")[:10], o.get("date", "")[:10]
            if d1 and d2 and abs((int(d1.replace("-", "")) - int(d2.replace("-", "")))) <= 1: return o
        except ValueError: pass
    return None

def is_savings(account_name, account_type):
    an = (account_name or "").upper()
    return account_type == "savings" or any(w in an for w in SAVINGS_WORDS)

def rule(r, p, rows_by_cust):
    """High-precision structural labels, or None to continue down the pipeline."""
    amt = r["amount"]; at = r.get("account_type", ""); k = p["kind"]; m = p["merchant"]
    if k == "empty": return "unknown"
    if k == "cash": return "transfer.cash"
    if k == "interest": return "income.capital" if amt > 0 else "unknown"
    if k == "salary": return "income.employment" if amt > 0 else "unknown"
    if k == "cardpayment_in": return "transfer.own" if amt > 0 else "unknown"
    if k == "transfer":
        if is_savings(r.get("account_name"), at): return "transfer.savings"
        cp = m.upper()
        if any(w in cp for w in SAVINGS_WORDS): return "transfer.savings"
        o = find_mirror(r, rows_by_cust)
        if o is not None:
            return "transfer.savings" if is_savings(o.get("account_name"), o.get("account_type")) else "transfer.own"
        if any(w in cp for w in EVERYDAY_WORDS): return "transfer.own"
        return "unknown"                       # no mirror row and counterpart not recognisable: abstain
    if k == "swish":
        msg = p["message"].strip()
        if not msg: return "transfer.person"
        for w in msg.split():
            if w in MSG_MAP: return MSG_MAP[w]
        return "unknown"                       # message present but not confidently mappable
    # --- merchant kind: fixed labels that depend on sign / account
    if re.search(r"\bCSN\b", m): return "credit.loan" if amt < 0 else "income.public"
    if at != "credit_card" and amt < 0 and RX_CARD_BILL.search(m): return "credit.loan"
    if at == "credit_card" and amt > 0 and RX_CARD_IN_ANY.search(m): return "transfer.own"
    if amt > 0 and any(s in m for s in SALES): return "income.sales"
    return None

# ---------------------------------------------------------------- LLM
SYSTEM = """You label Swedish bank transactions for a credit company. unknown is better than a wrong guess. Answer unknown unless you are at least 70% sure.
Return exactly one category per item from:
income.employment (salary from an employer), income.public (Försäkringskassan, CSN payout, pension, benefits), income.sales (selling own things: Tradera, Blocket, Vinted, Sellpy), income.capital (interest, dividends),
spend.housing (rent, landlord, BRF fee, electricity, broadband, mobile phone, home insurance, moving), spend.groceries (supermarkets, food shops), spend.dining (restaurants, cafés, fast food, food delivery), spend.transport (public transport, taxi, ride-hailing, scooters, fuel, parking, tolls), spend.shopping (clothes, beauty, home, furniture, electronics, DIY, online stores), spend.leisure (streaming, games, gym, cinema, events, flights, hotels, theme parks, travel), spend.health (pharmacy, doctor, dentist, patient fees), spend.family (childcare, school fees), spend.gambling (betting, casino, lotteries, deposits and winnings),
credit.loan (loan repayment, credit card bill, pay-later invoice, mortgage), credit.collections (debt collector, Kronofogden),
transfer.own, transfer.savings, transfer.person (private person, no purpose stated), transfer.cash, unknown.
Rules: Label by the kind of business, not the occasion (groceries abroad are groceries). KLARNA*/ZETTLE_*/SUMUP */IZ */PAYPAL * take the category of the shop behind them. A Klarna/Qliro/Walley invoice is credit.loan. Positive amount at a merchant is a refund and takes the merchant's category. A Swish to a private person takes the category its message names; with no clear purpose it is transfer.person. Use your knowledge of real Swedish merchants.
Output only a JSON array of {"i": index, "category": "...", "confidence": 0-1}."""

USAGE = Counter()

def _call_model(client, model, lines):
    resp = client.messages.create(
        model=model, max_tokens=4000,
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": lines}])
    u = getattr(resp, "usage", None)
    if u is not None:
        USAGE["calls"] += 1
        USAGE["input"] += getattr(u, "input_tokens", 0) or 0
        USAGE["output"] += getattr(u, "output_tokens", 0) or 0
        USAGE["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
        USAGE["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
    return "".join(getattr(b, "text", "") for b in resp.content)

def usage_cost():
    return sum(USAGE[k] * PRICE[k] / 1e6 for k in PRICE)

def _parse_llm(txt, valid_idx):
    """Tolerant parse -> {i: (cat, conf)}; malformed entries are simply absent (-> unknown)."""
    out = {}
    m = re.search(r"\[.*\]", txt or "", re.S)
    if not m: return out
    try: data = json.loads(m.group(0))
    except Exception:
        data = []
        for mm in re.finditer(r"\{[^{}]*\}", m.group(0)):
            try: data.append(json.loads(mm.group(0)))
            except Exception: pass
    for o in data if isinstance(data, list) else []:
        try:
            i = int(o.get("i")); c = str(o.get("category", "unknown")).strip(); conf = float(o.get("confidence", 0))
        except Exception:
            continue
        if i in valid_idx and c in CATS: out[i] = (c, max(0.0, min(1.0, conf)))
    return out

def llm_classify(items, model, retries=2, workers=4, batch_size=40):
    """items: dicts with i, text, hint -> {i: (cat, conf)}; a missing i means unknown."""
    if not items: return {}
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY not set; unresolved merchants become unknown", file=sys.stderr); return {}
    try:
        import anthropic
        client = anthropic.Anthropic(max_retries=0, timeout=60.0)
    except Exception as e:
        print(f"WARNING: anthropic client unavailable ({e}); unresolved merchants become unknown", file=sys.stderr); return {}
    from concurrent.futures import ThreadPoolExecutor
    batches = [items[s:s + batch_size] for s in range(0, len(items), batch_size)]

    def run(batch):
        lines = "\n".join(f'{it["i"]}. "{it["text"]}" ({it["hint"]})' for it in batch)
        valid = {it["i"] for it in batch}
        for attempt in range(retries + 1):
            try:
                return _parse_llm(_call_model(client, model, lines), valid)
            except Exception as e:
                if attempt < retries: time.sleep(2 * (2 ** attempt))
                else: print(f"LLM batch failed after {retries + 1} attempts ({type(e).__name__}): {str(e)[:120]} -> unknown", file=sys.stderr)
        return {}

    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(run, batches): out.update(res)
    return out

# ---------------------------------------------------------------- pipeline
def acct_class(r):
    at = r.get("account_type", "")
    return "card" if at == "credit_card" else ("savings" if is_savings(r.get("account_name"), at) else "checking")

def classify(rows, train_rows=None, model=DEFAULT_MODEL, cache_path="merchant_cache.json", conf_min=LLM_MIN_CONF, use_llm=True):
    """-> (preds {id: category}, sources {id: rule|train|exact|weak|llm|abstain})"""
    rows_by_cust = defaultdict(list)
    for r in rows: rows_by_cust[r.get("customer")].append(r)
    tmap = {}
    if train_rows:
        cnt = defaultdict(Counter)
        for r in train_rows:
            p = parse(r)
            if p["kind"] == "merchant" and r.get("category") in CATS:
                cnt[(merchant_key(p["merchant"]), r["amount"] > 0)][r["category"]] += 1
        for k, c in cnt.items():
            cat, n = c.most_common(1)[0]
            if n / sum(c.values()) >= 0.8: tmap[k] = cat
    cache = {}
    if os.path.exists(cache_path):
        try: cache = json.load(open(cache_path))
        except Exception: cache = {}

    preds, src, pending = {}, {}, {}
    def put(r, cat, source): preds[r["id"]] = cat; src[r["id"]] = source if cat != "unknown" else "abstain"
    for r in rows:
        p = parse(r); amt = r["amount"]
        c = rule(r, p, rows_by_cust)
        if c: put(r, c, "rule"); continue
        if p["kind"] == "swish":
            put(r, "unknown", "abstain"); continue
        key = merchant_key(p["merchant"])
        if not key.strip(): put(r, "unknown", "abstain"); continue
        if (key, amt > 0) in tmap: put(r, tmap[(key, amt > 0)], "train"); continue
        cat, strength = lex_match(key)
        if cat and strength == "exact" and (sign_ok(cat, amt) or cat.startswith("spend.")): put(r, cat, "exact"); continue   # positive = refund
        if cat and strength == "weak" and sign_ok(cat, amt): put(r, cat, "weak"); continue
        if amt >= SALARY_MIN and r.get("account_type") != "credit_card":
            # large deposit: salary only with a second signal, otherwise abstain
            put(r, "income.employment" if EMPLOYER_PAT.search(key) else "unknown", "rule"); continue
        ck = f"{key}|{'in' if amt > 0 else 'out'}|{acct_class(r)}"
        if ck in cache:
            cat, conf = cache[ck]; put(r, cat if conf >= conf_min else "unknown", "llm"); continue
        pending.setdefault(ck, []).append(r)

    if pending and use_llm:
        keys = list(pending); items = []
        for i, ck in enumerate(keys):
            rs = pending[ck]; key, sign, ac = ck.rsplit("|", 2)
            amts = [abs(x["amount"]) for x in rs]
            names = Counter(x.get("account_name", "") for x in rs).most_common(1)[0][0]
            hint = (f"money {sign}, account_type {rs[0].get('account_type') or ac}, account name '{names}', "
                    f"median {statistics.median(amts):.0f} SEK, {len(rs)} occurrence{'s' if len(rs) != 1 else ''}")
            items.append({"i": i, "text": key, "hint": hint})
        print(f"LLM: {len(items)} unique keys for {sum(len(v) for v in pending.values())} rows", file=sys.stderr)
        res = llm_classify(items, model)
        for i, ck in enumerate(keys):
            cat, conf = res.get(i, ("unknown", 0.0))
            cache[ck] = [cat, conf]
            for r in pending[ck]: put(r, cat if conf >= conf_min else "unknown", "llm")
        try: json.dump(cache, open(cache_path, "w"), ensure_ascii=False, indent=0)
        except Exception: pass
    for r in rows:
        if r["id"] not in preds: put(r, "unknown", "abstain")
    # sign sanity: income only positive; spend positive only when the merchant was recognised (refund)
    for r in rows:
        c, s, amt = preds[r["id"]], src[r["id"]], r["amount"]
        if c not in CATS: put(r, "unknown", "abstain")
        elif c.startswith("income.") and amt <= 0: put(r, "unknown", "abstain")
        elif c.startswith("spend.") and amt > 0 and s not in ("rule", "train", "exact") and c != "spend.gambling": put(r, "unknown", "abstain")
        elif c.startswith("credit.") and amt > 0: put(r, "unknown", "abstain")
    return preds, src

def score(rows, preds):
    n = len(rows); ok = sum(preds[r["id"]] == r["category"] for r in rows)
    unk = sum(preds[r["id"]] == "unknown" for r in rows); wrong = n - ok - unk
    return ok, wrong, unk, n

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("-o", "--out", default="predictions.csv")
    ap.add_argument("--train", default=os.environ.get("TRAIN_CSV", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.csv")))
    ap.add_argument("--no-train-map", action="store_true"); ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--model", default=os.environ.get("CLASSIFIER_MODEL", DEFAULT_MODEL))
    ap.add_argument("--conf", type=float, default=LLM_MIN_CONF)
    ap.add_argument("--cache", default=os.environ.get("MERCHANT_CACHE", "merchant_cache.json"))
    a = ap.parse_args()
    t0 = time.time()
    rows = load_csv(a.input)
    train = load_csv(a.train) if (os.path.exists(a.train) and not a.no_train_map) else None
    preds, src = classify(rows, train, model=a.model, cache_path=a.cache, conf_min=a.conf, use_llm=not a.no_llm)
    out_rows = [(r["id"], preds[r["id"]]) for r in rows]
    assert len(out_rows) == len(rows), "output row count must equal input"
    assert all(c in CATS for _, c in out_rows), "every output row must carry a valid category"
    assert len({i for i, _ in out_rows}) == len(out_rows), "duplicate ids in input"
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "category"])
        for i, c in out_rows: w.writerow([i, c])
    print(f"wrote {a.out}: {len(rows)} rows in {time.time()-t0:.1f}s | sources {dict(Counter(src.values()))}", file=sys.stderr)
    print(f"LLM usage: {dict(USAGE)} cost=${usage_cost():.4f}", file=sys.stderr)
    if rows and "category" in rows[0]:
        ok, wrong, unk, n = score(rows, preds)
        print(f"score: correct={ok} wrong={wrong} unknown={unk} n={n}  acc={ok/n:.3f}  net={(ok-wrong)/n:.3f}", file=sys.stderr)
        for r in rows:
            if preds[r["id"]] != r["category"]:
                print(f"  MISS {r['id']} {r['amount']:>10} {r['text'][:45]:45} true={r['category']} pred={preds[r['id']]} ({src[r['id']]})", file=sys.stderr)
