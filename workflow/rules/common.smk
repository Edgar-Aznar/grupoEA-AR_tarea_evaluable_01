from snakemake.utils import min_version

min_version("9.0")

# Valores del config que usan varias reglas
SIZES = config["subsampling"]["sizes"]
SEEDS = config["subsampling"]["seeds"]
LIBRARIES = list(config["enrichment"]["libraries"])
DIRECTIONS = ["up","down"]
