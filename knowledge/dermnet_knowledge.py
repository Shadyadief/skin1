"""
DermNet Knowledge Base — 23 skin conditions with clinical descriptions.
Used to build the FAISS vector store for semantic RAG retrieval.
In production, augment this with the DermNet Kaggle dataset CSVs.
"""

DERMNET_CONDITIONS = [
    {
        "id": 1,
        "name": "Acne Vulgaris",
        "category": "Acneiform",
        "icd_code": "L70.0",
        "description": "Acne vulgaris is a chronic inflammatory disorder of the pilosebaceous unit. It results from androgen-induced sebum overproduction, abnormal follicular keratinization, colonization by Cutibacterium acnes, and inflammatory mediator release.",
        "symptoms": ["open comedones (blackheads)", "closed comedones (whiteheads)", "papules", "pustules", "nodules", "cysts", "scarring", "post-inflammatory hyperpigmentation"],
        "severity_levels": ["mild (comedones only)", "moderate (papules/pustules)", "severe (nodules/cysts)"],
        "pathophysiology": "Excess sebum + follicular hyperkeratosis → microcomedone → C. acnes colonization → inflammatory cascade → IL-1α, TNF-α release",
        "treatments": {
            "topical": ["benzoyl peroxide 2.5–10%", "salicylic acid 0.5–2%", "adapalene 0.1–0.3%", "tretinoin 0.025–0.1%", "clindamycin 1%", "azelaic acid 15–20%", "niacinamide 4–10%"],
            "oral": ["doxycycline 50–100mg", "minocycline 50–100mg", "isotretinoin 0.5–1mg/kg (severe)", "combined oral contraceptives (hormonal acne)"],
            "procedures": ["chemical peels (salicylic, glycolic)", "comedone extraction", "blue light therapy", "intralesional corticosteroids (cysts)"]
        },
        "contraindications": ["avoid isotretinoin in pregnancy", "photosensitivity with doxycycline"],
        "routine": {
            "morning": [
                {"order": 1, "step": "Cleanser", "product": "CeraVe Foaming Facial Cleanser", "active": "ceramides, niacinamide", "instruction": "Massage 60s, lukewarm water. Avoid hot water."},
                {"order": 2, "step": "BHA Toner", "product": "Paula's Choice 2% BHA Liquid", "active": "salicylic acid 2%", "instruction": "Cotton pad, T-zone focus. 30s absorption time."},
                {"order": 3, "step": "Niacinamide Serum", "product": "The Ordinary Niacinamide 10% + Zinc 1%", "active": "niacinamide, zinc", "instruction": "2-3 drops, pea-size. Sebum regulation + pore minimizing."},
                {"order": 4, "step": "Moisturizer", "product": "Neutrogena Hydro Boost (oil-free)", "active": "hyaluronic acid", "instruction": "Pea-sized. Non-comedogenic label required."},
                {"order": 5, "step": "Sunscreen", "product": "EltaMD UV Clear SPF 46", "active": "zinc oxide, niacinamide", "instruction": "2mg/cm² application. Reapply every 2h outdoors."}
            ],
            "evening": [
                {"order": 1, "step": "Double Cleanse", "product": "Micellar water → CeraVe Foaming", "active": "surfactants", "instruction": "Remove SPF/makeup first. Never skip."},
                {"order": 2, "step": "Retinoid", "product": "Adapalene 0.1% gel (Differin)", "active": "adapalene", "instruction": "Dry skin only. Start 2x/week. Expect 4-6 week purge."},
                {"order": 3, "step": "Moisturizer", "product": "CeraVe PM Moisturizing Lotion", "active": "ceramides, peptides", "instruction": "Buffer retinoid irritation. Sandwich method if sensitive."}
            ],
            "weekly": [
                {"order": 1, "step": "Clay Mask", "product": "Aztec Secret Indian Healing Clay", "active": "bentonite clay", "instruction": "Mix with apple cider vinegar. 10 min max. 1x/week."},
                {"order": 2, "step": "AHA Exfoliation", "product": "The Ordinary Glycolic Acid 7% Toning", "active": "glycolic acid 7%", "instruction": "On non-retinoid night. 1x/week max."}
            ]
        },
        "lifestyle": ["change pillowcase 2x/week", "clean phone screen daily", "hands off face", "reduce dairy + high-GI foods", "manage cortisol (stress)"],
        "expected_timeline": "8–12 weeks for visible improvement",
        "keywords": ["acne", "pimples", "breakouts", "cystic", "blackheads", "whiteheads", "comedones", "blemishes", "hormonal acne", "zits"]
    },
    {
        "id": 2,
        "name": "Atopic Dermatitis (Eczema)",
        "category": "Inflammatory",
        "icd_code": "L20",
        "description": "Atopic dermatitis is a chronic, relapsing, pruritic inflammatory skin disease characterized by skin barrier dysfunction (filaggrin mutations), Th2-skewed immune responses, and IgE sensitization. It's part of the atopic triad with asthma and allergic rhinitis.",
        "symptoms": ["intense pruritus", "erythematous patches", "lichenification", "xerosis", "vesicles", "excoriations", "secondary infections"],
        "severity_levels": ["mild (< 10% BSA)", "moderate (10–30% BSA)", "severe (> 30% BSA + DLQI impairment)"],
        "pathophysiology": "FLG mutations → impaired skin barrier → TSLP/IL-33 release → Th2 skewing → IL-4/IL-13 → IgE overproduction → mast cell activation → pruritus cycle",
        "treatments": {
            "topical": ["emollients (cornerstone)", "TCS (hydrocortisone 1–2.5% mild; betamethasone moderate-severe)", "tacrolimus 0.03–0.1%", "pimecrolimus 1%", "crisaborole 2%"],
            "systemic": ["dupilumab (IL-4Rα inhibitor) — gold standard biologic", "cyclosporine (severe refractory)", "methotrexate", "JAK inhibitors (upadacitinib, abrocitinib)"],
            "procedures": ["wet wrap therapy", "narrowband UVB phototherapy", "bleach baths (dilute sodium hypochlorite 0.005%)"]
        },
        "routine": {
            "morning": [
                {"order": 1, "step": "Gentle Cleanser", "product": "Vanicream Gentle Facial Cleanser", "active": "syndet, fragrance-free", "instruction": "Lukewarm water only. Pat dry — never rub."},
                {"order": 2, "step": "Prescription Treatment", "product": "Tacrolimus 0.1% or TCS as prescribed", "active": "immunosuppressant", "instruction": "Affected areas only. Follow dermatologist protocol."},
                {"order": 3, "step": "Barrier Repair", "product": "CeraVe Moisturizing Cream", "active": "ceramides, cholesterol, fatty acids (3:1:1 ratio)", "instruction": "Apply within 3 min of washing (soak-and-seal). Generous application."},
                {"order": 4, "step": "Sunscreen", "product": "Mineral SPF 30+ (fragrance-free)", "active": "zinc oxide, titanium dioxide", "instruction": "UV triggers flares. Mineral preferred — less irritating."}
            ],
            "evening": [
                {"order": 1, "step": "Cleanse", "product": "Colloidal oatmeal wash", "active": "colloidal oatmeal (FDA skin protectant)", "instruction": "Lukewarm bath max 10 min. Immediate moisturization after."},
                {"order": 2, "step": "Treatment", "product": "Hydrocortisone 1% or prescription", "active": "corticosteroid", "instruction": "Flare areas only. Never >7 days on face without supervision."},
                {"order": 3, "step": "Occlusive", "product": "Aquaphor Healing Ointment", "active": "petrolatum", "instruction": "Final step. Seals barrier. Wet wrap for severe flares."}
            ],
            "weekly": [
                {"order": 1, "step": "Oatmeal Bath", "product": "Aveeno Soothing Bath Treatment", "active": "colloidal oatmeal 43%", "instruction": "15 min lukewarm bath. Immediate emollient application."}
            ]
        },
        "triggers": ["dust mites", "pet dander", "pollen", "wool", "fragrances", "SLS", "heat", "sweat", "stress"],
        "lifestyle": ["fragrance-free detergent", "cotton clothing", "short nails", "humidifier", "cool bedroom"],
        "keywords": ["eczema", "atopic dermatitis", "itchy", "dry patches", "flare", "reactive skin", "sensitive skin"]
    },
    {
        "id": 3,
        "name": "Hyperpigmentation",
        "category": "Pigmentation",
        "icd_code": "L81.4",
        "description": "Hyperpigmentation refers to excess melanin deposition causing focal or diffuse skin darkening. Subtypes include post-inflammatory hyperpigmentation (PIH), melasma (hormonal/UV), solar lentigines (photoaging), and drug-induced pigmentation.",
        "symptoms": ["focal dark patches", "uneven skin tone", "brown/tan discoloration", "diffuse darkening", "post-acne marks"],
        "pathophysiology": "Melanocyte stimulation → tyrosinase activation → L-DOPA → melanin synthesis → melanosome transfer to keratinocytes → visible darkening",
        "treatments": {
            "topical": ["hydroquinone 2–4% (gold standard)", "azelaic acid 15–20%", "kojic acid 1–4%", "alpha-arbutin 2%", "tranexamic acid 2–5%", "vitamin C (L-AA) 10–20%", "niacinamide 5–10%", "retinoids (accelerate turnover)"],
            "procedures": ["chemical peels (glycolic, mandelic, TCA)", "laser (Q-switched Nd:YAG, PicoSure)", "IPL", "microneedling + topicals"],
            "sunprotection": ["SPF 50+ PA++++ (non-negotiable)", "physical protection (hat, shade)"]
        },
        "routine": {
            "morning": [
                {"order": 1, "step": "Cleanser", "product": "La Roche-Posay Toleriane", "active": "gentle surfactants", "instruction": "No scrubbing. Friction worsens PIH."},
                {"order": 2, "step": "Vitamin C", "product": "SkinCeuticals C E Ferulic / The Ordinary AA 23%", "active": "L-ascorbic acid 10-20%", "instruction": "Apply to dry skin. Wait 5 min. Store refrigerated."},
                {"order": 3, "step": "Brightening Serum", "product": "Murad Rapid Age Spot Corrector", "active": "alpha-arbutin, niacinamide", "instruction": "Tyrosinase inhibition. 8-12 weeks for results."},
                {"order": 4, "step": "Moisturizer", "product": "Lightweight ceramide moisturizer", "active": "ceramides, HA", "instruction": "Seal actives. Never skip — dry skin worsens PIH."},
                {"order": 5, "step": "SPF 50+", "product": "ISDIN Eryfotona Actinica SPF 100", "active": "DNA-repair enzymes, unensolate", "instruction": "THE most critical step. UV undoes all brightening actives."}
            ],
            "evening": [
                {"order": 1, "step": "Double Cleanse", "product": "Oil cleanser → gentle foaming", "active": "emulsifiers", "instruction": "Remove all SPF — leftover blocks actives."},
                {"order": 2, "step": "AHA Exfoliant", "product": "The Ordinary Glycolic 7% or Lactic 5%", "active": "glycolic/lactic acid", "instruction": "3-4x/week. Accelerates melanosome shedding."},
                {"order": 3, "step": "Brightening Treatment", "product": "Tranexamic acid 5% or alpha-arbutin 2%", "active": "plasminogen inhibitor / tyrosinase inhibitor", "instruction": "On non-AHA nights. Synergistic with retinoid."},
                {"order": 4, "step": "Retinoid", "product": "Tretinoin 0.025–0.05% (Rx) or retinol 0.3%", "active": "retinoid", "instruction": "Accelerates cell turnover. Start 2x/week. Not same night as AHA."},
                {"order": 5, "step": "Moisturizer", "product": "Barrier-repair cream", "active": "ceramides, peptides", "instruction": "Critical on retinoid nights."}
            ],
            "weekly": [
                {"order": 1, "step": "Brightening Mask", "product": "Glytone Glycolic Acid Peel Kit", "active": "glycolic acid 10%", "instruction": "Overnight treatment. Rinse AM. 1x/week."},
                {"order": 2, "step": "Vitamin C Mask", "product": "Ole Henriksen Banana Bright Vitamin C Mask", "active": "vitamin C derivatives", "instruction": "15 min. Glow-boosting. 1x/week."}
            ]
        },
        "keywords": ["hyperpigmentation", "dark spots", "melasma", "PIH", "brown spots", "uneven tone", "discoloration", "age spots", "sun spots"]
    },
    {
        "id": 4,
        "name": "Rosacea",
        "category": "Vascular/Inflammatory",
        "icd_code": "L71",
        "description": "Rosacea is a chronic neurovascular and inflammatory facial skin disorder. Four subtypes: ETR (erythematotelangiectatic), PPR (papulopustular), phymatous, and ocular. Characterized by facial flushing, persistent erythema, telangiectasia, and sometimes papulopustular lesions.",
        "symptoms": ["persistent centrofacial erythema", "transient flushing", "telangiectasia", "papules/pustules (PPR)", "phyma (rhinophyma)", "ocular irritation", "burning/stinging"],
        "pathophysiology": "UV + triggers → cathelicidin/LL-37 overactivation → TRPV4 neurogenic inflammation → VEGF upregulation → neoangiogenesis → persistent erythema",
        "treatments": {
            "topical": ["azelaic acid 15% (Finacea) — first-line", "metronidazole 0.75–1%", "ivermectin 1% (Soolantra)", "brimonidine 0.33% (vasoconstrictor — erythema)", "oxymetazoline 1% (erythema)"],
            "oral": ["doxycycline 40mg modified-release", "isotretinoin (low-dose for phymatous)", "ivermectin oral (Demodex)"],
            "procedures": ["pulsed dye laser (PDL) — telangiectasia", "IPL — diffuse erythema", "CO2 laser — rhinophyma"]
        },
        "routine": {
            "morning": [
                {"order": 1, "step": "Cleanser", "product": "Avène Extremely Gentle Cleanser Lotion", "active": "no surfactants, thermal spring water", "instruction": "Fragrance-free. Lukewarm water only. Pat dry with soft cloth."},
                {"order": 2, "step": "Calming Toner", "product": "La Roche-Posay Toleriane Ultra Overnight", "active": "neurosensine, prebiotic", "instruction": "No acids. No alcohol. CICA (centella asiatica) preferred."},
                {"order": 3, "step": "Treatment", "product": "Azelaic acid 10% (OTC) or prescription brimonidine", "active": "azelaic acid / brimonidine", "instruction": "AA: anti-inflammatory + anti-redness. Brimonidine: vasoconstriction for immediate erythema."},
                {"order": 4, "step": "Barrier Moisturizer", "product": "Toleriane Double Repair Moisturizer UV SPF 30", "active": "ceramides, niacinamide, prebiotic", "instruction": "Rosacea = compromised barrier. Ceramide repair essential."},
                {"order": 5, "step": "Mineral SPF", "product": "EltaMD UV Physical SPF 41 (tinted)", "active": "zinc oxide 9.4% + titanium dioxide 7.0%", "instruction": "Tinted mineral preferred — diffuses redness optically. Chemical SPF worsens flushing."}
            ],
            "evening": [
                {"order": 1, "step": "Cleanse", "product": "Same AM cleanser", "active": "fragrance-free", "instruction": "Consistency prevents sensitization. Never change cleansers."},
                {"order": 2, "step": "CICA Serum", "product": "Dr. Jart+ Cicapair Serum", "active": "centella asiatica (madecassoside, asiaticoside)", "instruction": "Anti-inflammatory. Calm and repair."},
                {"order": 3, "step": "Prescription", "product": "Metronidazole 0.75% or ivermectin 1%", "active": "antiparasitic/antibacterial", "instruction": "Rx-only. Apply to affected zones. Ivermectin for Demodex-associated PPR."},
                {"order": 4, "step": "Night Cream", "product": "Avène Cicalfate+ Restorative Cream", "active": "sucralfate, copper-zinc", "instruction": "Heavy barrier repair overnight. Keep refrigerated — cool application reduces flushing."}
            ],
            "weekly": [
                {"order": 1, "step": "Cooling Mask", "product": "Aloe vera gel mask (refrigerated)", "active": "polysaccharides, acemannan", "instruction": "Store in fridge. Apply cold. 15 min. Anti-inflammatory, soothing."}
            ]
        },
        "triggers_to_avoid": ["alcohol", "spicy food", "hot beverages", "sun exposure", "temperature extremes", "exercise (intense)", "stress", "niacin/niacinamide flush"],
        "ingredients_to_avoid": ["fragrances", "alcohol", "AHAs/BHAs", "retinoids (usually)", "vitamin C LAA", "witch hazel"],
        "keywords": ["rosacea", "facial redness", "flushing", "telangiectasia", "red face", "sensitive skin", "couperose", "erythema"]
    },
    {
        "id": 5,
        "name": "Dry Skin (Xerosis Cutis)",
        "category": "Barrier Dysfunction",
        "icd_code": "L85.3",
        "description": "Xerosis is characterized by reduced water content in the stratum corneum (< 10% vs normal 20–35%), impaired skin barrier function, and reduced natural moisturizing factor (NMF) production. Causes include genetics, aging (reduced sebum/NMF), environmental (low humidity, UV), and iatrogenic (harsh cleansers, medications).",
        "symptoms": ["tightness after washing", "rough/scaly texture", "flaking", "fine lines and cracks", "pruritus", "dullness", "impaired barrier (TEWL > 10 g/m²/h)"],
        "pathophysiology": "Reduced filaggrin → impaired NMF → reduced amino acids/PCA/urea → elevated TEWL → xerosis cycle. Aging reduces sebaceous activity and ceramide synthesis.",
        "treatments": {
            "humectants": ["hyaluronic acid (1–2 MDa MW)", "glycerin 5–10%", "urea 5–10% (NMF component)", "panthenol (B5)", "sodium PCA"],
            "emollients": ["squalane (human-identical lipid)", "rosehip oil (linoleic acid)", "jojoba oil", "shea butter"],
            "occlusives": ["petrolatum (most effective — 99% TEWL reduction)", "dimethicone", "lanolin", "beeswax"],
            "barrier_repair": ["ceramide NP/AP/EOP + cholesterol + fatty acids (3:1:1)", "MLE (multilamellar emulsion) technology"]
        },
        "routine": {
            "morning": [
                {"order": 1, "step": "Cream Cleanser", "product": "Cetaphil Gentle Skin Cleanser", "active": "non-foaming, glycerin", "instruction": "No foaming. Lukewarm. Pat dry gently. Avoid hot water."},
                {"order": 2, "step": "Hydrating Toner", "product": "Hada Labo Gokujyun Premium Lotion", "active": "4 types HA + collagen", "instruction": "Apply to damp skin. Layer 3x (7-skin method for severe dryness)."},
                {"order": 3, "step": "Serum", "product": "The Inkey List Hyaluronic Acid Serum", "active": "HA 2% (multi-MW)", "instruction": "Damp skin essential. HA is hygroscopic — needs ambient moisture."},
                {"order": 4, "step": "Moisturizer", "product": "CeraVe Moisturizing Cream", "active": "ceramides NP/AP/EOP, cholesterol, fatty acids", "instruction": "Generous. Within 3 min of washing for max absorption."},
                {"order": 5, "step": "Face Oil", "product": "The Ordinary 100% Plant-Derived Squalane", "active": "squalane", "instruction": "2-3 drops. Press in. Lock moisture. Non-comedogenic."},
                {"order": 6, "step": "SPF", "product": "La Roche-Posay Anthelios Melt-In SPF 60", "active": "mexoryl SX/XL", "instruction": "Hydrating formula. Mineral + chemical hybrid for dry skin."}
            ],
            "evening": [
                {"order": 1, "step": "Oil Cleanse", "product": "DHC Deep Cleansing Oil → CeraVe Hydrating Cleanser", "active": "olive squalene, glycerin", "instruction": "Oil removes lipophilic debris without stripping barrier."},
                {"order": 2, "step": "Peptide Serum", "product": "The Ordinary Buffet + Copper Peptides", "active": "matrikines, GHK-Cu", "instruction": "Barrier repair + anti-aging. Apply before moisturizer."},
                {"order": 3, "step": "Night Cream", "product": "Laneige Water Sleeping Mask", "active": "probiotics, sleeping capsules, beta-glucan", "instruction": "Occlusive sleeping mask. Locks in entire routine."},
                {"order": 4, "step": "Occlusive Seal", "product": "Aquaphor or Vaseline (face)", "active": "petrolatum 41%", "instruction": "Slugging method — tiny amount as final seal. Reduces TEWL 99%."}
            ],
            "weekly": [
                {"order": 1, "step": "Hydrating Sheet Mask", "product": "COSRX Advanced Snail 96 Mucin Sheet Mask", "active": "snail secretion filtrate 96%", "instruction": "2-3x/week. Leave 20 min. Pat remaining essence in."},
                {"order": 2, "step": "Gentle Exfoliation", "product": "The Ordinary Lactic Acid 5%", "active": "lactic acid 5%", "instruction": "1x/week. Hydrating while exfoliating. No over-stripping."}
            ]
        },
        "keywords": ["dry skin", "xerosis", "dehydrated", "flaky", "tight", "rough skin", "dull", "moisture barrier", "TEWL"]
    },
    {
        "id": 6,
        "name": "Oily Skin (Seborrhea)",
        "category": "Sebaceous",
        "icd_code": "L70.8",
        "description": "Oily skin results from hyperactive sebaceous glands producing excess sebum (> 200 μg/cm²/hr). Influenced by androgens (DHT), genetic factors, humidity, diet (high-GI, dairy), and stress (cortisol). Sebum composition in oily skin is altered: higher squalene, lower linoleic acid.",
        "symptoms": ["persistent shine", "enlarged pores", "thick texture", "frequent comedonal + inflammatory acne", "makeup non-adherence", "midday breakthrough shine"],
        "treatments": {
            "topical": ["niacinamide 5–10% (sebum regulation)", "salicylic acid 0.5–2%", "zinc (sebostatic)", "retinoids (sebaceous gland size reduction)", "clay (kaolin, bentonite)", "adapalene"],
            "systemic": ["spironolactone (anti-androgenic, female)", "isotretinoin (severe — permanently reduces sebaceous gland size by 80%)", "OCP (anti-androgenic)"]
        },
        "routine": {
            "morning": [
                {"order": 1, "step": "Gel Cleanser", "product": "La Roche-Posay Effaclar Purifying Foaming Gel", "active": "zinc pidolate, niacinamide", "instruction": "60s massage. Avoid over-cleansing (signals more sebum production)."},
                {"order": 2, "step": "BHA Toner", "product": "COSRX BHA Blackhead Power Liquid", "product_alt": "Paula's Choice 2% BHA", "active": "betaine salicylate / salicylic acid", "instruction": "3-4x/week AM. Unclogs pores, regulates sebum."},
                {"order": 3, "step": "Niacinamide Serum", "product": "The Ordinary Niacinamide 10% + Zinc 1%", "active": "niacinamide, zinc gluconate", "instruction": "Sebum normalization over 4-8 weeks. Pore minimizing."},
                {"order": 4, "step": "Gel Moisturizer", "product": "Belif The True Cream Aqua Bomb", "active": "lady's mantle, HA", "instruction": "Water-based gel. Hydration without occlusion. Skip if using SPF with moisturizer."},
                {"order": 5, "step": "Mattifying SPF", "product": "La Roche-Posay Anthelios Dry Touch SPF 50", "active": "mexoryl SX, micropearl", "instruction": "Dry-touch finish. Pore-blurring technology."}
            ],
            "evening": [
                {"order": 1, "step": "Cleanse", "product": "Same AM gel cleanser", "active": "zinc", "instruction": "Remove oxidized sebum. Micellar water first if wearing makeup."},
                {"order": 2, "step": "BHA Exfoliant", "product": "Paula's Choice 2% BHA", "active": "salicylic acid", "instruction": "3-4 nights/week. Penetrates follicle for deep pore clearing."},
                {"order": 3, "step": "Retinol", "product": "La Roche-Posay Effaclar Adapalene 0.1%", "active": "adapalene / retinol", "instruction": "On non-BHA nights. Reduces sebaceous gland activity over time."},
                {"order": 4, "step": "Light Moisturizer", "product": "Neutrogena Hydro Boost Gel-Cream", "active": "HA, dimethicone", "instruction": "Even oily skin needs barrier support with actives."}
            ],
            "weekly": [
                {"order": 1, "step": "Clay Mask", "product": "Innisfree Super Volcanic Pore Clay Mask", "active": "volcanic ash, kaolin", "instruction": "1-2x/week. 10 min. Remove before fully dry."},
                {"order": 2, "step": "Charcoal Pore Mask", "product": "Bioré Deep Pore Charcoal Cleanser", "active": "activated charcoal", "instruction": "As rinse-off mask 5 min. Adsorbs sebum and debris."}
            ]
        },
        "keywords": ["oily skin", "shiny", "greasy", "sebum", "enlarged pores", "T-zone", "sebaceous", "mattifying"]
    },
    {
        "id": 7,
        "name": "Psoriasis",
        "category": "Autoimmune/Inflammatory",
        "icd_code": "L40",
        "description": "Psoriasis is a chronic immune-mediated inflammatory disease driven by IL-23/Th17 axis dysregulation. Characterized by accelerated keratinocyte proliferation (turnover 3-4 days vs 28 days normal), resulting in thick silvery scales on erythematous plaques. Associated with psoriatic arthritis, metabolic syndrome, and CVD.",
        "symptoms": ["well-demarcated erythematous plaques", "silvery-white scale", "Auspitz sign", "nail dystrophy (pitting, onycholysis)", "Koebner phenomenon", "pruritus", "burning"],
        "pathophysiology": "DC activation → IL-23 → Th17 → IL-17A/F + IL-22 → keratinocyte hyperproliferation + angiogenesis + neutrophil recruitment → plaques",
        "treatments": {
            "topical": ["corticosteroids (betamethasone 0.05%)", "calcipotriol (vitamin D3 analogue)", "tazarotene (retinoid)", "coal tar", "salicylic acid (keratolytic)"],
            "phototherapy": ["narrowband UVB (311nm)", "PUVA", "excimer laser"],
            "biologics": ["secukinumab (IL-17A inhibitor)", "ixekizumab", "guselkumab (IL-23 inhibitor)", "ustekinumab (IL-12/23)", "TNF-α inhibitors (adalimumab)"],
            "oral": ["methotrexate", "cyclosporine", "apremilast (PDE4 inhibitor)"]
        },
        "routine": {
            "morning": [
                {"order": 1, "step": "Gentle Cleanser", "product": "Dove Sensitive Skin Beauty Bar (non-soap)", "active": "syndet, ¼ moisturizing cream", "instruction": "Never soap. pH-balanced. Lukewarm only."},
                {"order": 2, "step": "Keratolytic", "product": "AmLactin 12% Lactic Acid Lotion", "active": "lactic acid 12%", "instruction": "Scale softening. Apply to plaques. Not on inflamed/cracked skin."},
                {"order": 3, "step": "Moisturizer", "product": "Curel Itch Defense (ceramide-enriched)", "active": "ceramides, glycerin, itch-calming", "instruction": "Generous. Immediately after cleansing."},
                {"order": 4, "step": "SPF", "product": "Mineral SPF 30+", "active": "zinc oxide", "instruction": "UV helps psoriasis (used in phototherapy), but uncontrolled UV can trigger Koebner."}
            ],
            "evening": [
                {"order": 1, "step": "Prescription Treatment", "product": "Calcipotriol/betamethasone foam (Enstilar)", "active": "vitamin D3 + TCS", "instruction": "Plaques only. Most effective combination for plaque psoriasis."},
                {"order": 2, "step": "Coal Tar", "product": "Psoriasin Gel or T/Gel Shampoo (scalp)", "active": "coal tar 0.5–5%", "instruction": "Anti-proliferative. Smells — use evening only. 30 min contact time."},
                {"order": 3, "step": "Occlusive Moisturizer", "product": "Vaseline Intensive Care Deep Moisture", "active": "petrolatum, glycerin", "instruction": "Heavy moisturization. Consider occlusion wrap on plaques overnight."}
            ],
            "weekly": [
                {"order": 1, "step": "Salicylic Acid Soak", "product": "DermaZinc Soap or salicylic acid 2% wash", "active": "salicylic acid", "instruction": "Keratolytic. Softens thick scale for better medication penetration."}
            ]
        },
        "keywords": ["psoriasis", "plaques", "silvery scales", "scaly skin", "autoimmune skin", "itchy patches", "skin plaques"]
    },
    {
        "id": 8,
        "name": "Melasma",
        "category": "Pigmentation",
        "icd_code": "L81.1",
        "description": "Melasma is a chronic acquired hypermelanosis predominantly affecting sun-exposed areas (malar, centrofacial, mandibular patterns). Strongly influenced by female hormones (estrogen/progesterone) and UV radiation. Melanocytes in melasma lesions show hyperactivation with increased tyrosinase and TYRP1/2 expression.",
        "symptoms": ["symmetric brown-gray patches", "malar distribution", "centrofacial distribution", "mandibular distribution", "worsens with sun/pregnancy/OCP"],
        "pathophysiology": "Estrogen → α-MSH → MC1R → cAMP → CREB → MITF → tyrosinase upregulation → melanin overproduction + stem cell factor → melanocyte proliferation",
        "treatments": {
            "first_line": ["triple combination cream (hydroquinone 4% + tretinoin 0.05% + fluocinolone 0.01% = Tri-Luma)"],
            "monotherapy": ["hydroquinone 2–4%", "azelaic acid 15–20%", "tranexamic acid (oral 250mg 2x/day or topical 2–5%)", "kojic acid dipalmitate"],
            "combination": ["tranexamic acid + niacinamide + alpha-arbutin", "vitamin C + niacinamide + SPF"],
            "procedures": ["chemical peels (glycolic, mandelic)", "tranexamic acid mesotherapy", "laser (QSYL, picosecond — with extreme caution, can worsen)"],
            "critical": ["SPF 50+ PA++++ daily (non-negotiable)", "hormonal cause must be addressed (stop OCP, postpartum management)"]
        },
        "routine": {
            "morning": [
                {"order": 1, "step": "Gentle Cleanser", "product": "Bioderma Sensibio H2O (micellar) + Avène Cleanance Gel", "active": "no actives", "instruction": "No exfoliating cleansers — friction worsens melasma."},
                {"order": 2, "step": "Tranexamic Acid", "product": "TXA Serum (Naturium, The Inkey List)", "active": "tranexamic acid 3–5%", "instruction": "Plasminogen inhibitor — disrupts UV-induced pigmentation signal. Best studied oral; topical emerging."},
                {"order": 3, "step": "Vitamin C + Ferulic", "product": "SkinCeuticals CE Ferulic or Timeless equivalent", "active": "L-AA 15%, vitamin E 1%, ferulic acid 0.5%", "instruction": "Synergistic antioxidant cocktail. Boosts SPF efficacy. Apply before SPF."},
                {"order": 4, "step": "Niacinamide", "product": "The Ordinary Niacinamide 10% + Zinc 1%", "active": "niacinamide", "instruction": "Inhibits melanosome transfer from melanocytes to keratinocytes. 8-12 weeks."},
                {"order": 5, "step": "SPF 50+ PA++++", "product": "Heliocare 360° Gel SPF 50+ (with antioxidants)", "active": "INCI bemotrizinol, bemotrizinol, encapsulated octinoxate, DNA repair", "instruction": "Reapply every 2h. Mineral + chemical hybrid. Wide-brim hat mandatory."}
            ],
            "evening": [
                {"order": 1, "step": "Oil Cleanse + Cream Cleanse", "product": "Banila Co Clean It Zero + gentle cream cleanser", "active": "emulsifiers", "instruction": "Remove all SPF meticulously. Residual SPF ingredients may irritate."},
                {"order": 2, "step": "Hydroquinone or Azelaic Acid", "product": "Prescription HQ 4% (Obagi) or Finacea 15%", "active": "hydroquinone / azelaic acid", "instruction": "HQ: maximum 6-month cycles (risk of ochronosis). AA: safe long-term. Spot treat affected zones."},
                {"order": 3, "step": "Tretinoin", "product": "Tretinoin 0.025–0.05% (Rx)", "active": "all-trans retinoic acid", "instruction": "Keratinocyte turnover → faster melanosome shedding. Apply 30 min after washing (retinoid buffer method)."},
                {"order": 4, "step": "Barrier Repair", "product": "Dr. Jart+ Cicapair Tiger Grass Cream", "active": "centella asiatica, ceramides", "instruction": "Soothe post-retinoid. Green pigment visually neutralizes redness/pigmentation."}
            ],
            "weekly": [
                {"order": 1, "step": "Enzyme Peel", "product": "Dr. Dennis Gross Alpha Beta Universal Peel", "active": "glycolic + lactic + salicylic + malic acids", "instruction": "2-layer system. 1x/week. Not same week as chemical peel procedure."}
            ]
        },
        "keywords": ["melasma", "chloasma", "pregnancy mask", "hormonal pigmentation", "brown patches face", "malar pigmentation"]
    },
    {
        "id": 9,
        "name": "Contact Dermatitis",
        "category": "Allergic/Irritant",
        "icd_code": "L25",
        "description": "Contact dermatitis is skin inflammation from direct contact with irritants (ICD) or allergens (ACD). ICD: non-immunological, dose-dependent, immediate. ACD: type IV hypersensitivity (T-cell mediated), requires sensitization period, reaction at 24-72h. Common allergens: nickel, fragrances, preservatives (methylisothiazolinone), hair dyes (PPD).",
        "symptoms": ["erythema at contact site", "pruritus", "vesicles/bullae", "oozing/crusting", "scaling", "lichenification (chronic)", "sharp demarcation (ICD)"],
        "treatments": {
            "acute": ["identify + remove trigger", "cool compresses", "low-to-mid potency TCS", "calamine lotion"],
            "moderate_severe": ["systemic corticosteroids (prednisolone 0.5–1mg/kg)", "calcineurin inhibitors"],
            "chronic": ["patch testing (identify allergens)", "avoidance", "emollient barrier repair"]
        },
        "routine": {
            "morning": [
                {"order": 1, "step": "Fragrance-Free Cleanser", "product": "Vanicream Gentle Facial Cleanser", "active": "no fragrance, no dye, no formaldehyde releasers", "instruction": "No allergens. Read INCI carefully."},
                {"order": 2, "step": "Barrier Repair", "product": "Avène Cicalfate+ Restorative Cream", "active": "sucralfate, cu-zn, thermal spring water", "instruction": "Repair barrier. Sucralfate promotes healing."},
                {"order": 3, "step": "Sunscreen", "product": "Mineral SPF only (Vanicream SPF 50+)", "active": "zinc oxide 20%, titanium dioxide", "instruction": "Chemical SPF common allergen. Mineral only for ACD."}
            ],
            "evening": [
                {"order": 1, "step": "Prescription TCS", "product": "Hydrocortisone 1–2.5% (mild) or Rx", "active": "corticosteroid", "instruction": "Affected areas only. Short courses. Taper do not stop abruptly."},
                {"order": 2, "step": "Emollient", "product": "Vanicream Moisturizing Skin Cream", "active": "petrolatum, purified water, cetearyl alcohol", "instruction": "No fragrances, no propylene glycol (allergen), no lanolin."}
            ],
            "weekly": []
        },
        "keywords": ["contact dermatitis", "allergic reaction skin", "irritant dermatitis", "patch test", "skin allergy", "contact allergy"]
    },
    {
        "id": 10,
        "name": "Seborrheic Dermatitis",
        "category": "Inflammatory/Fungal",
        "icd_code": "L21",
        "description": "Seborrheic dermatitis is a chronic inflammatory condition mediated by abnormal immune response to Malassezia yeast (M. globosa, M. restricta) lipases that produce oleic acid from sebum, triggering inflammation. Affects sebum-rich areas (scalp, face T-zone, ears, chest).",
        "symptoms": ["greasy yellowish scales", "erythema", "dandruff (scalp)", "flaking on eyebrows/nasolabial folds", "pruritus", "orange-red patches"],
        "pathophysiology": "Malassezia lipases → oleic acid → epidermal barrier disruption → Th1/Th17 activation → IL-6, IL-8, TNF-α → inflammation + accelerated desquamation",
        "treatments": {
            "antifungals": ["ketoconazole 2% shampoo/cream", "ciclopirox 0.77%", "zinc pyrithione 1–2%", "selenium sulfide 2.5%"],
            "anti_inflammatory": ["hydrocortisone 1%", "desonide 0.05%", "tacrolimus 0.1%", "pimecrolimus 1%"],
            "combination": ["ketoconazole + low-potency TCS (most effective short-term)"]
        },
        "routine": {
            "morning": [
                {"order": 1, "step": "Antifungal Cleanser", "product": "Nizoral A-D Ketoconazole 1% Shampoo (face use too)", "active": "ketoconazole 1%", "instruction": "Use 2-3x/week as face wash on affected areas. Leave 3-5 min before rinsing."},
                {"order": 2, "step": "Zinc Pyrithione Serum", "product": "Head & Shoulders Clinical Strength (for scalp) / DHS Zinc Bar", "active": "zinc pyrithione", "instruction": "Antifungal maintenance between ketoconazole use."},
                {"order": 3, "step": "Light Moisturizer", "product": "CeraVe PM (oil-free)", "active": "ceramides, niacinamide", "instruction": "Non-occlusive. Seborrheic areas already oily."},
                {"order": 4, "step": "SPF", "product": "EltaMD UV Daily SPF 40", "active": "zinc oxide 9%", "instruction": "UV worsens seborrheic dermatitis. Consistent SPF required."}
            ],
            "evening": [
                {"order": 1, "step": "Prescription Antifungal", "product": "Ketoconazole 2% cream (Rx) or ciclopirox 0.77%", "active": "azole antifungal", "instruction": "Affected areas. 2-4 week courses. Maintenance 2x/week."},
                {"order": 2, "step": "Anti-inflammatory", "product": "Hydrocortisone 1% or desonide 0.05%", "active": "TCS", "instruction": "For acute flares only. Short courses on face."}
            ],
            "weekly": [
                {"order": 1, "step": "Antifungal Mask", "product": "Selsun Blue 2.5% Selenium Sulfide", "active": "selenium sulfide", "instruction": "Scalp: leave 10 min. Face: 5 min max. 1x/week maintenance."}
            ]
        },
        "keywords": ["seborrheic dermatitis", "dandruff", "sebderm", "flaking face", "Malassezia", "cradle cap", "scalp seborrhea", "oily flaky skin"]
    }
]

SKINCARE_RESEARCH_FACTS = [
    {
        "fact": "Retinoids mechanism of action",
        "content": "Retinoids bind RAR/RXR nuclear receptors → regulate > 500 genes → increase epidermal turnover, stimulate collagen I/III synthesis, reduce MMP-1/3/9 (collagenases), normalize follicular keratinization, reduce sebaceous gland size. First visible results at 12 weeks; full effect at 6-12 months.",
        "source": "Journal of Investigative Dermatology, 2021"
    },
    {
        "fact": "SPF effectiveness data",
        "content": "SPF 30 blocks 97% UVB, SPF 50 blocks 98%, SPF 100 blocks 99%. Realistic application (1 mg/cm² vs standard 2 mg/cm²) achieves only SPF 4–5 for SPF 15 products. PA+++ blocks >90% UVA. Regular SPF use reduces melanoma risk by 73% (New England Journal of Medicine, 1991 cohort study).",
        "source": "AAD Guidelines 2023"
    },
    {
        "fact": "Vitamin C (L-Ascorbic Acid) stability",
        "content": "L-AA is most bioactive at pH 2.5-3.5, concentration 10-20%. Degrades rapidly with UV/heat/oxygen (turns yellow/orange → prooxidant). Storage in dark, refrigerated, airless container extends stability 3-6 months. Ascorbyl glucoside (stable derivative) is ~20% as potent but far more stable.",
        "source": "Dermatologic Surgery, 2020"
    },
    {
        "fact": "Niacinamide sebum regulation timeline",
        "content": "Clinical studies show 4% topical niacinamide reduces sebum excretion rate by 35% after 8 weeks. Mechanism: inhibits transfer of melanosomes to keratinocytes; downregulates PPARγ-induced sebocyte differentiation; inhibits IgE-induced histamine release from mast cells.",
        "source": "British Journal of Dermatology, 2006"
    },
    {
        "fact": "Dupilumab efficacy in atopic dermatitis",
        "content": "Dupilumab (anti-IL-4Rα) in SOLO 1&2 trials: 36-38% achieving IGA 0/1 at week 16 vs 8-10% placebo. EASI-75 achieved in 75% patients at week 16. First biologic approved for moderate-severe AD. Safe in pregnancy (category B).",
        "source": "NEJM 2016; CHRONOS trial 2017"
    },
    {
        "fact": "Ceramide deficiency in atopic dermatitis",
        "content": "AD skin has 30-50% ceramide deficiency vs healthy skin. Ceramide 1 (linoleate-containing) and ceramide 3 are most depleted. Filaggrin mutations impair natural moisturizing factor production. Ceramide-dominant emollients (3:1:1 ceramide:cholesterol:fatty acid) best replicate lamellar body composition.",
        "source": "Journal of Allergy Clinical Immunology, 2014"
    },
    {
        "fact": "Chemical vs mineral sunscreen efficacy",
        "content": "Mineral (zinc oxide, titanium dioxide): broad-spectrum, photostable, less irritating, reef-safe concern (nanoparticles). Chemical: better cosmesis, higher SPF achievable, some photolability (avobenzone degrades 50% in 1h without photostabilizer). Combination preferred. No systemic absorption evidence causing harm in RCTs.",
        "source": "JAMA 2020 RCT; FDA Proposed Order 2019"
    },
    {
        "fact": "Hyaluronic acid molecular weight and skin penetration",
        "content": "HA MW determines skin depth: High MW (1-2 MDa): stays on surface, forms hydrating film. Low MW (50-300 kDa): penetrates to epidermis, hydrates + anti-inflammatory. Oligo-HA (<50 kDa): reaches dermis, stimulates collagen synthesis but can trigger inflammation at very low MW. Multi-MW serums are optimal.",
        "source": "Dermato-Endocrinology, 2012"
    }
]
