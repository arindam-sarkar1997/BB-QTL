"""Plot labels for the 78 pathways in pathway_descriptions.csv.

Labels derive from column 2. The missing PWY3O-450 description is supplied
from the previously verified SGD name, phosphatidylcholine biosynthesis I.
These are display labels, not replacements for pathway identifiers.

Abbreviations: synth. = biosynthesis; degr. = degradation;
super = superpathway; FA = fatty acid; PA = phosphatidate;
PC = phosphatidylcholine; G3P = glycerol-3-phosphate;
DHAP = dihydroxyacetone phosphate; THF = tetrahydrofolate;
ACMS = 2-amino-3-carboxymuconate semialdehyde;
SAM = S-adenosyl-L-methionine; Dol-P = dolichyl phosphate;
ETC = electron transport chain; PI = phosphatidylinositol;
4-HB = 4-hydroxybenzoate. Amino acids use three-letter codes.
For compact display, nucleotide labels omit 'de novo'; the original
descriptions retain this detail. 'Deoxy' distinguishes deoxyribonucleotides.
"""

pathway_shorthands = {
    "PWY3O-19": "Ubiquinol-6 synth. (4-HB)",
    "PWY-922": "Mevalonate I",
    "PWY3O-242": "PI phosphate synth.",
    "PWY3O-94": "TCA & glyoxylate cycles",
    "PWY3O-64": "Met salvage",
    "PWY3O-862": "Ubiquinone synth. (super)",
    "PWY3O-48": "Glycerol synth.",
    "PWY-781": "Sulfate assimilation",
    "PWY3O-6499": "PA synth. II (G3P)",
    "PWY3O-6407": "PA synth. I (DHAP)",
    "PWY3O-45": "Folate synth. II",
    "PWY3O-6635": "PA synth. (super)",
    "PWY3O-188": "Aerobic respiration (ETC)",
    "PWY-5651": "L-Trp degr. to ACMS",
    "PWY3O-351": "Met salvage (super)",
    "PWY3O-1743": "Mannose degr.",
    "PWY3O-7": "Thr synth. (super)",
    "PWY-5686-1": "UMP synth. I",
    "PWY3O-261": "Ser & Gly synth.",
    "PWY3O-3827": "Glucose-6-P synth.",
    "PWY3O-355": "Stearate synth. III",
    "PWY3O-5962": "FA synth. (sat. & unsat.)",
    "PWY3O-4": "Carnitine shuttle",
    "PWY-5971-1": "Myristate synth.",
    "PWY3O-4158": "NAD synth. (super)",
    "PWY3O-450": "PC synth. I",
    "PWY3O-123": "Dol-P-mannose synth.",
    "PWY3O-4107": "NAD salvage V (PNC V)",
    "PWY3O-2": "Phospholipid synth. (super)",
    "PWY3O-4153": "Phe synth.",
    "PWY-6074-1": "Zymosterol synth.",
    "PWY-6123-1": "IMP synth. I",
    "PWY3O-10": "FA synth. (initial steps)",
    "PWY3O-13": "Glu synth. (super)",
    "PWY3O-8514": "Palmitate synth.",
    "PWY3O-6336": "FA synth. (saturated)",
    "PWY-5694": "Allantoin to glyoxylate I",
    "PWY-6126-1": "Adenosine synth. II (super)",
    "PWY0-162": "Pyrimidine nucleotide synth.",
    "PWY3O-4109": "Ile degr.",
    "PWY3O-4112": "Leu degr.",
    "PWY-7220-1": "Deoxyadenosine nucleotide synth. II",
    "PWY3O-4105": "Val degr.",
    "PWY3O-69": "Heme & siroheme synth.",
    "PWY-7222-1": "Deoxyguanosine nucleotide synth. II",
    "PWY-821-1": "Sulfur amino acid synth.",
    "PWY-6075-1": "Ergosterol synth. I",
    "PWY3O-4031": "Glycogen synth.",
    "PWY-6482-1": "Diphthamide synth.",
    "PWY3O-954": "Met synth. (super)",
    "PWY-5653": "NAD synth. from ACMS",
    "PWY-7219": "Adenosine nucleotide synth.",
    "PWY-7221": "Guanosine nucleotide synth.",
    "PWY3O-285": "Purine synth. & salvage",
    "PWY0-662": "PRPP synth.",
    "PWY-5084": "2-Oxoglutarate to succinyl-CoA",
    "PWY-7176": "UTP & CTP synth.",
    "PWY-6125": "Guanosine synth. II (super)",
    "PWY3O-4108": "L-Tyr degr. III",
    "PWY3O-981": "Acetoin & butanediol synth.",
    "PWY3O-31704": "Ergosterol synth.",
    "PWY3O-2220": "Adenine/hypoxanthine salvage",
    "PWY-2201": "Folate transformations I",
    "PWY3O-4120": "Tyr synth.",
    "PWY-6614": "THF synth.",
    "PWY3O-1": "Purine & nucleoside salvage",
    "PWY3O-214": "Trp degr.",
    "PWY3O-50": "Tetrapyrrole synth.",
    "PWY3O-20": "Folate polyglutamylation",
    "PWY3O-4115": "Phe degr.",
    "PWY3O-440": "Pyruvate to acetoin III",
    "PWY3O-259": "Phospholipids II (Kennedy)",
    "PWY3O-697": "Folate interconversions",
    "PWY-5760-1": "Beta-alanine synth. IV",
    "PWY-5041": "SAM cycle II",
    "PWY3O-4300": "Ethanol degr.",
    "PWY3O-1565": "Dol-P-glucose synth.",
    "PWY3O-15": "Chitin synth.",
}

# Optional caption text for figures using these labels.
abbreviation_caption = (
    "PI, phosphatidylinositol; PC, phosphatidylcholine; PA, phosphatidate; "
    "FA, fatty acid; 4-HB, 4-hydroxybenzoate; "
    "ACMS, 2-amino-3-carboxymuconate semialdehyde; "
    "G3P, glycerol-3-phosphate; DHAP, dihydroxyacetone phosphate; "
    "THF, tetrahydrofolate; SAM, S-adenosylmethionine; "
    "Dol-P, dolichyl phosphate; ETC, electron transport chain; "
    "synth., biosynthesis; degr., degradation; super, superpathway. "
    "Amino acids use three-letter codes. Full pathway names and IDs "
    "should accompany abbreviated labels."
)
