DATA_DIR={$1}

wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/cocostuff-10k-v1.1.zip -P $DATA_DIR
unzip $DATA_DIR/cocostuff-10k-v1.1.zip -d $DATA_DIR
