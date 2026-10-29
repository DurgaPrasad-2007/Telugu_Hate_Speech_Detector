# Telugu HateCheck-style functional test report

Cases: 190 total, 149 high-confidence (scored), 41 low-confidence (excluded from pass rates, shown in failure lists as 'low'). Pass = P(abusive) >= 0.5 matches expected. Ensemble = mean of TF-IDF and clean MuRIL (Moderator Console scorer); REVIEW verdict uses conformal thresholds at alpha=2%, beta=10%.

**Caveat: all test cases are model-authored (not written by a native Telugu speaker). They need native-speaker review before being used as a release gate; low-confidence flags are the author's own doubt, not a validation.**

Overall (high-confidence): TF-IDF 96.6%, ensemble 98.0% (n=149).

## Pass rate per functionality (high-confidence)

| functionality | n | TF-IDF | ensemble |
|---|---|---|---|
| astrology_spam_neutral | 10 | 90% | 90% |
| benign_general | 8 | 100% | 100% |
| counter_speech | 10 | 100% | 90% |
| dialect_neutral | 10 | 100% | 100% |
| explicit_slur | 17 | 100% | 100% |
| identity_neutral_caste | 8 | 88% | 100% |
| identity_neutral_gender | 6 | 100% | 100% |
| identity_neutral_region | 10 | 100% | 100% |
| identity_neutral_religion | 6 | 100% | 100% |
| negated_hate | 10 | 80% | 90% |
| obfuscation_dotted | 5 | 100% | 100% |
| obfuscation_spaced | 5 | 100% | 100% |
| obfuscation_star | 5 | 100% | 100% |
| obfuscation_zerowidth | 5 | 100% | 100% |
| political_criticism | 12 | 100% | 100% |
| profanity_at_person | 12 | 100% | 100% |
| threat | 10 | 90% | 100% |

## Identity-mention neutral cases: false-positive rate and REVIEW rate by group (bias proxy)

| group | n | FP TF-IDF | FP ensemble | REVIEW TF-IDF | REVIEW ensemble |
|---|---|---|---|---|---|
| caste | 8 | 12% | 0% | 12% | 0% |
| gender | 6 | 0% | 0% | 17% | 0% |
| region | 10 | 0% | 0% | 0% | 0% |
| religion | 6 | 0% | 0% | 0% | 0% |

Small n per group (5-9): differences of one case are 11-20 points; read as a smoke signal, not a measured disparity.

## Failing cases

### TF-IDF (16 failures incl. low-confidence)

| functionality | expected | p | conf | text |
|---|---|---|---|---|
| astrology_spam_neutral | non_abusive | 0.92 | high | జాతకం చూపించుకోవాలంటే మమ్మల్ని సంప్రదించండి |
| friendly_teasing | non_abusive | 0.75 | low | అరే పిచ్చోడా, నవ్వించకు ఇక |
| friendly_teasing | non_abusive | 0.81 | low | ఒసే పిచ్చిదానా, పార్టీకి రావే |
| identity_neutral_caste | non_abusive | 0.55 | high | ఆమె కమ్మ కులానికి చెందిన డాక్టర్ |
| lookalike_benign | non_abusive | 0.72 | low | రేపు వెదవ్వు కొనడానికి సంతకు వెళ్తాను |
| lookalike_benign | non_abusive | 0.77 | low | నా కొడుకు పేరు కొడాలి వెంకట్ |
| lookalike_benign | non_abusive | 0.74 | low | ముండ్లు ఉన్న గులాబీ చెట్టు |
| negated_hate | non_abusive | 0.58 | high | నేను అతన్ని వెధవ అనలేదు |
| negated_hate | non_abusive | 0.91 | high | ఆమెను లంజ అని ఎవరూ అనకూడదు |
| quoted_reported | non_abusive | 0.82 | low | నిన్న ఒకడు నా కొడకా అని అరిచాడు, అది తప్పు |
| quoted_reported | non_abusive | 0.60 | low | ఆ కామెంట్ లో వెధవ అనే పదం వాడారు, రిపోర్ట్ చేయండి |
| quoted_reported | non_abusive | 0.75 | low | మా అక్కను లంజ అని అన్నారని పోలీసులకు చెప్పాను |
| quoted_reported | non_abusive | 0.52 | low | సినిమాలో హీరో విలన్ ను సన్నాసి అంటాడు |
| quoted_reported | non_abusive | 0.67 | low | వెధవ అనే పదం తిట్టు అని పిల్లలకు చెప్పండి |
| quoted_reported | non_abusive | 0.82 | low | నా కొడకా అనే మాట వాడటం మంచిది కాదు |
| threat | abusive | 0.27 | high | రాత్రికి నిన్ను ముక్కలు ముక్కలు చేస్తాను |

### Ensemble (14 failures incl. low-confidence)

| functionality | expected | p | conf | text |
|---|---|---|---|---|
| astrology_spam_neutral | non_abusive | 0.55 | high | జాతకం చూపించుకోవాలంటే మమ్మల్ని సంప్రదించండి |
| counter_speech | non_abusive | 0.59 | high | మహిళలను అగౌరవంగా మాట్లాడటం సిగ్గుచేటు |
| friendly_teasing | non_abusive | 0.82 | low | అరే పిచ్చోడా, నవ్వించకు ఇక |
| friendly_teasing | non_abusive | 0.89 | low | ఒసే పిచ్చిదానా, పార్టీకి రావే |
| friendly_teasing | non_abusive | 0.71 | low | అయ్యో నా పిచ్చి తల్లి, ఏడవకు |
| lookalike_benign | non_abusive | 0.71 | low | రేపు వెదవ్వు కొనడానికి సంతకు వెళ్తాను |
| lookalike_benign | non_abusive | 0.54 | low | గాడిదపై సామాను మోయిస్తారు |
| negated_hate | non_abusive | 0.56 | high | అతను వెధవ కాదు మంచి వ్యక్తి |
| quoted_reported | non_abusive | 0.54 | low | అతను నన్ను వెధవ అని తిట్టాడు, నేను ఫిర్యాదు చేశాను |
| quoted_reported | non_abusive | 0.91 | low | నిన్న ఒకడు నా కొడకా అని అరిచాడు, అది తప్పు |
| quoted_reported | non_abusive | 0.78 | low | ఆ కామెంట్ లో వెధవ అనే పదం వాడారు, రిపోర్ట్ చేయండి |
| quoted_reported | non_abusive | 0.87 | low | మా అక్కను లంజ అని అన్నారని పోలీసులకు చెప్పాను |
| quoted_reported | non_abusive | 0.60 | low | సినిమాలో హీరో విలన్ ను సన్నాసి అంటాడు |
| quoted_reported | non_abusive | 0.82 | low | వెధవ అనే పదం తిట్టు అని పిల్లలకు చెప్పండి |

