After some concerns about the Parquet file, I've uploaded the same data as a Pickle file too to alleviate any issues. This should play nicer with Pandas. Feel free to use either.

If you want to stick with the Parquet file, but are running into a loading error - try loading it using pyarrow version 4.0.1:

`pip install pyarrow==4.0.1`