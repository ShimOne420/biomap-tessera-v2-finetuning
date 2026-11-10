# Scopo: benchmark V1/V2 a pooling denso con supporto pixel identico.
# Fasi: griglia target -> aggregazione V1/V2 -> appaiamento -> CV spaziale -> report.
# Input: tile V1 int8, embedding V2 fp32, raster GLOBIO MSA.
# Output: feature per cella, predizioni OOF, metriche appaiate e report Markdown.
# Parametri/esempi: scripts/dense_msa.py --help.
# Risorse: CPU; GPU/MPS solo per completare tile V2 privi di embedding.
# Ripresa: gli embedding V2 sono scritti con memmap e progresso atomico per tile.
