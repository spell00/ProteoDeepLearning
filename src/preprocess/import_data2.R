library(MSnbase)
library(stringr)
library(xlsx)
library(optparse)
# Create an intensity matrix from a list of MS spectra
# spectra_files must be in mzXML format
import_sp <- function(spectra_file, sample_num, binSize, msLevel = 2) {
  get_label <- function (sp_data, sample_num) {

    assays_names <- ls(assayData(sp_data))
    label <- str_split(spectra_file, '/')
    label <- label[[1]][length(label[[1]])]

    label <- str_split(str_split(label, 'RD151_')[[1]][2], "_inj")[[1]][[1]]

    i <- 1
    for (assay_name in assays_names) {
      # sample_num <- str_split(str_split(assay_name[[1]], '.S')[[1]][1], 'F')[[1]][2]
      # sample_num <- strtoi(gsub("(?<![0-9])0+", "", sample_num, perl = TRUE))
      assays_names[[i]] <- paste0(label, "-", sample_num)
      i <- i + 1
    }

    return(assays_names)
  }

  sp_data <- readMSData(spectra_file, msLevel=msLevel, verbose=TRUE)
  # TIC (total ion count) > 1e4
  # sp_data <- sp_data[tic(sp_data) > 1e4]
  labels <- get_label(sp_data, sample_num)
  bins <- bin(sp_data, binSize=binSize)
  tmp <- mz(bins)
  list_names <- names(assayData(bins))
  cols <- tmp[[list_names[1]]]
  bined_sp <- do.call(rbind, MSnbase::intensity(bin(sp_data, binSize=binSize)))
  tics <- tic(bins)
  rts <- rtime(bins)
  if (msLevel == 2) {
    precursorMzs <- precursorMz(bins)
    bined_sp <- cbind(precursorMzs, tics, rts, bined_sp)
    row.names(bined_sp) <- labels
    cols <- c("precursorMz", "tics", "rtime", cols)
  } else {
    bined_sp <- cbind(tics, rts, bined_sp)
    row.names(bined_sp) <- labels
    cols <- c("tics", "rtime", cols)
 }

  colnames(bined_sp) <- cols

  return(bined_sp)
}
# writeMSData(
#   bined_sp,
#   'test',
#   outformat = "mzml",
# )

option_list <- list(
  make_option("--outIntensities", type="character", default="inputs_bin1/RD151",
              help="Output files for intensities", metavar="character"),
  make_option("--dir", type="character", default= "RD151_210625/mzml/",
              help="spectro directory name", metavar="character"),
  make_option("--bin", type="integer", default= "1", metavar="character"),
  make_option("--g", type="character", default= "90", metavar="character")
);

opt_parser <- OptionParser(option_list=option_list);
opt <- parse_args(opt_parser);

spectra_files <- list.files(paste0(opt$dir,"g",opt$g), pattern = '*.mzML',full.names = TRUE)

i <- 1
for (spectra_file in spectra_files) {
  # arr <- str_split(opt$outIntensities, "/")[[1]]
  intensity_matrix <- import_sp(spectra_file, sample_num=i, binSize = opt$bin, msLevel = 1)
  # spectra_file <- paste(arr[1], 'm1', arr[2], sep = '/')
  label <- str_split(str_split(spectra_file, 'RD151_')[[1]][3], "_inj")[[1]][[1]]
  filename <- paste0(opt$outIntensities, "_", label, "_bin", opt$bin, "_m1_", i,  ".csv")
  # file_sp <- file(filename, open="wt")
  write.table(intensity_matrix, filename, sep=",", row.names=TRUE)
  i <- i + 1
}


i <- 1
for (spectra_file in spectra_files) {
  # arr <- str_split(opt$outIntensities, "/")[[1]]
  intensity_matrix <- import_sp(spectra_file, sample_num=i, binSize = opt$bin, msLevel = 2)
  # spectra_file <- paste(arr[1], 'm2', arr[2], sep = '/')
  label <- str_split(str_split(spectra_file, 'RD151_')[[1]][3], "_inj")[[1]][[1]]
  filename <- paste0(opt$outIntensities, "_", label, "_bin", opt$bin, "_m2_", i,  ".csv")
  # file_sp <- file(filename, open="wt")
  write.table(intensity_matrix, filename, sep=",", row.names=TRUE)
  i <- i + 1
}



# rtime(toto$F1.S10001)
# MSpectra(ass$F1.S00001)
# sp2 <- new("Spectrum2", mz = c(1, 2, 3, 4), intensity = c(5, 3, 2, 5), precursorMz = 2)
# writeMzTabData()
# spl <- MSpectra(sp1, sp2)
# Bin the intensities values according to the bin size

