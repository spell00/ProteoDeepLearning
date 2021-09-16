library(MSnbase)
library(stringr)
library(xlsx)
library(optparse)
# Create an intensity matrix from a list of MS spectra
# spectra_files must be in mzXML format
import_sp <- function(spectra_files, binSize) {
  get_labels <- function (sp_data) {

    assays_names <- ls(assayData(sp_data))
    labels <- str_split(spectra_files, '/')

    i <- 1
    for (label in labels) {
      # TODO look for labels in sp_data instead
      # label <- label[[1]]
      labels[[i]] <- str_split(str_split(label[length(label)], 'RD151_')[[1]][2], "_inj")[[1]][[1]]
      i <- i + 1
    }

    i <- 1
    for (assay_name in assays_names) {
      sample_num <- str_split(str_split(assay_name[[1]], '.S')[[1]][1], 'F')[[1]][2]
      sample_num <- strtoi(gsub("(?<![0-9])0+", "", sample_num, perl = TRUE))
      assays_names[[i]] <- paste0(labels[sample_num][[1]], "-", sample_num)
      i <- i + 1
    }

    return(assays_names)
  }

  # Use spectra only with MS level 1
  sp_data <- readMSData(spectra_files, msLevel=2, verbose=TRUE)
  # TIC (total ion count) > 1e4
  sp_data <- sp_data[tic(sp_data)> 1e4]
  labels <- get_labels(sp_data)
  path <- paste0(labels[1:length(labels)-1], collapse = "/")

  for (i in 1:length(labels)) {
    label <- split(labels[i], "\\")
    label[length(label) - 1] <- paste0(label[length(label) - 1],'ticked',collapse = '_')
    label <- label[[length(label)]]
    writeMSData(sp_data[i], paste0(path, label,"_binSize",binSize,'.mzml', collapse = ''))
  }
  # Bin the intensities values according to the bin size
  bined_sp <- do.call(rbind, MSnbase::intensity(bin(sp_data, binSize=binSize)))
  row.names(bined_sp) <- labels
  return(bined_sp)
}

option_list <- list(
  make_option("--outIntensities", type="character", default="inputs_bin1/RD151",
              help="Output files for intensities", metavar="character"),
  make_option("--dir", type="character", default= "RD151_210625/mzml/",
              help="spectro directory name", metavar="character"),
  make_option("--g", type="character", default= "30", metavar="character"),
  make_option("--bin", type="character", default= "1", metavar="character")
);

opt_parser <- OptionParser(option_list=option_list);
opt <- parse_args(opt_parser);

spectra_files <- list.files(paste0(opt$dir,"g",opt$g), pattern = '*.mzML',full.names = TRUE)
intensity_matrix <- import_sp(spectra_files, opt$bin)

file_sp <- file(paste0(opt$outIntensities, "_g", opt$g,".csv"), open="wt")

# rownames(intensity_matrix) <- labels
write.table(intensity_matrix, file_sp, sep=",", row.names=TRUE)

