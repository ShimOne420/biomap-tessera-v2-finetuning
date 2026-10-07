# Fonti, licenze e riferimenti

Il codice nuovo è MIT. Le licenze dei dati e dei pesi sono separate: MIT non concede diritti sui dataset.

- TESSERA: https://github.com/ucam-eo/tessera, commit `e368539f7c4711e5f16fdd09801573830e34baf9`. I file `student.py` e `reference.py` conservano codice ufficiale MIT (Frank Feng, 2025); in `reference.py` sono adattati soltanto gli import di package. Licenza originale in `TESSERA_LICENSE.txt`.
- Medium: https://huggingface.co/geotessera/TESSERA-V-2.0-2B-M, revisione `8c80b926f67cb0b44c2cd41d0226e61741503b9a`; SHA256 `3823be7db9d9cfc93f3c2a47c7699be82821ab4e1117d4d2befdb746941ee96e`. Pesi scaricati separatamente; consultare la model card per termini e provenienza.
- Paper: https://arxiv.org/abs/2607.03949. Medium è uno student del teacher 2B; il repository non scarica il teacher.
- GLOBIO MSA 2020: https://www.globio.info/globioweb. Target overall `MSA-area / cell-area`, griglia 10 arcsec. Conservare readme/licenza della sorgente; i raster non sono redistribuiti nel repository.
- NHM BII v2.1.1: https://data.nhm.ac.uk/dataset/bii-developed-by-nhm-v2-1-1-limited-release, DOI https://doi.org/10.5519/k33reyb6, CC-BY-NC-SA-4.0. Raster 2020, 5 arcmin, percentuale divisa per 100. Scaricare implica i termini NHM. Dataset destinato a uso non commerciale; artefatti derivati vanno valutati sotto i termini della sorgente prima di redistribuirli.
- Sentinel-1/2: https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc e https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a. Cataloghi e dati Copernicus con attribuzioni/termini delle collezioni. S1 RTC MPC è gamma intensity: conversione `10 log10(power)` equivalente a `20 log10(sqrt(power))`. Non riusare automaticamente la vecchia conversione applicata all'intensità come se fosse ampiezza.

`configs/provenance.json` identifica i run V1/V2 precedenti e gli hash dei dataset originari. I CSV pubblicati contengono geometrie e split scelti per il progetto, senza target NHM/GLOBIO né embedding. Questi split sono stati già esaminati e non costituiscono un test finale cieco.
