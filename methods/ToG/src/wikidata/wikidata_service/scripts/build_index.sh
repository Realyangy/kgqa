#!/bin/bash

cd "$(dirname "$0")/../.." || exit 1

for i in {0..9}; do
    python -u -m wikidata_service.database.db_deploy.build_index --input_dir /dev/shm/wikidump_inmem/wikidump_20230116 --num_chunks 10 --chunk_idx $i --output_dir /dev/shm/wikidump_inmem/wikidump_20230116/indices > logs/build_index_${i}.log 2>&1 &
done

wait
