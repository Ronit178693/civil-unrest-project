"""
lexicon.py
==========
Civil-unrest domain lexicon: trigger phrases grouped by event type, plus
light gazetteer lists used as features for argument-role classification.
Kept separate from the statistical model so it can be extended without
retraining (a common pattern in event-extraction systems for low-resource
domains: lexicon-based candidate generation + learned disambiguation).
"""

TRIGGER_LEXICON = {
    "PROTEST": ["protest", "protests", "protesting", "rally", "rallies",
                "demonstration", "demo", "march", "marches", "marching",
                "procession", "agitation"],
    "STRIKE": ["strike", "strikes", "walkout", "walk-out", "shutdown",
               "bandh", "hartal", "boycott"],
    "CLASH": ["clash", "clashes", "clashed", "confrontation", "scuffle",
              "standoff", "skirmish"],
    "RIOT": ["riot", "riots", "unrest", "mob", "looting", "rampage"],
    "SIT_IN": ["sit-in", "sit in", "occupation", "blockade", "road block",
               "dharna"],
    "REPRESSION": ["arrested", "detained", "crackdown", "tear gas",
                   "lathi charge", "curfew", "baton charge", "water cannon"],
}

# flat lookup: surface form (lowercased) -> event type
FLAT_TRIGGERS = {w: etype for etype, words in TRIGGER_LEXICON.items() for w in words}

ESCALATION_WORDS = {"riot", "riots", "mob", "looting", "rampage", "tear gas",
                     "lathi charge", "baton charge", "water cannon", "clash",
                     "clashes", "clashed"}

ACTOR_HINTS = ["students", "farmers", "workers", "unions", "activists",
               "residents", "teachers", "youth", "opposition", "community",
               "communities", "groups", "protesters", "demonstrators"]

SCALE_WORDS = {"dozen", "dozens", "hundreds", "thousand", "thousands",
               "handful", "group", "crowd"}

# Words that typically open or make up a time expression (TIME argument).
TIME_WORDS = {"today", "tonight", "tomorrow", "yesterday", "now", "morning", "evening",
              "afternoon", "night", "overnight", "weekend", "week", "month", "monday", "tuesday",
              "wednesday", "thursday", "friday", "saturday", "sunday", "am", "pm", "hours",
              "days", "weeks", "since", "last", "next"}

HEDGE_WORDS = {"reportedly", "allegedly", "unconfirmed", "claims", "rumored",
               "rumoured"}  # evasive / moderation-avoidant language signal

# Gazetteer of Indian place names (major cities + states/UTs), used as a CRF
# feature for LOCATION. Deliberately much broader than the 16 cities in the
# synthetic data, so the feature encodes "is a known place name" rather than
# memorising the training cities. Extend with GeoNames for real deployments.
GAZETTEER = {p.lower() for p in [
    "Delhi", "New Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru", "Bangalore", "Hyderabad",
    "Ahmedabad", "Pune", "Surat", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane",
    "Bhopal", "Visakhapatnam", "Patna", "Vadodara", "Ghaziabad", "Ludhiana", "Agra", "Nashik",
    "Faridabad", "Meerut", "Rajkot", "Varanasi", "Srinagar", "Aurangabad", "Dhanbad", "Amritsar",
    "Allahabad", "Prayagraj", "Ranchi", "Howrah", "Coimbatore", "Jabalpur", "Gwalior",
    "Vijayawada", "Jodhpur", "Madurai", "Raipur", "Kota", "Guwahati", "Chandigarh", "Mysuru",
    "Mysore", "Noida", "Gurugram", "Gurgaon", "Bhubaneswar", "Dehradun", "Shimla", "Jammu",
    "Thiruvananthapuram", "Kochi", "Mangaluru", "Imphal", "Shillong", "Aizawl", "Kohima",
    "Agartala", "Gangtok", "Itanagar", "Panaji", "Puducherry", "Silchar", "Siliguri", "Udaipur",
    "Ajmer", "Bikaner", "Jhansi", "Gorakhpur", "Bareilly", "Aligarh", "Moradabad", "Saharanpur",
    "Muzaffarpur", "Gaya", "Bhagalpur", "Cuttack", "Rourkela", "Jamshedpur", "Bokaro",
    "Bihar", "Punjab", "Haryana", "Kerala", "Assam", "Manipur", "Gujarat", "Maharashtra",
    "Karnataka", "Telangana", "Odisha", "Rajasthan", "Uttarakhand", "Jharkhand", "Kashmir",
    "Ladakh", "Goa", "Tripura", "Nagaland", "Mizoram", "Meghalaya", "Sikkim",
    "Tamil Nadu", "West Bengal", "Uttar Pradesh", "Madhya Pradesh", "Andhra Pradesh",
    "Himachal Pradesh", "Arunachal Pradesh", "Chhattisgarh",
]}
