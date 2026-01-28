#!/bin/bash

set -e
cd ../blossom

BASE_CMD="python3 main.py"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

TOTAL_EXPERIMENTS=18
CURRENT_EXPERIMENT=0

run_experiment() {
    local partitioner=$1
    local aggregation=$2
    local audio_clients=$3
    local image_clients=$4
    local both_clients=$5
    
    CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Experiment ${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS}${NC}"
    echo -e "${BLUE}Partitioner: ${partitioner}${NC}"
    echo -e "${BLUE}Aggregation: ${aggregation}${NC}"
    echo -e "${BLUE}Clients: audio=${audio_clients}, image=${image_clients}, both=${both_clients}${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    CMD="${BASE_CMD} \
        dataset=avmnist \
        partitioner=${partitioner} \
        aggregation=${aggregation} \
        experiment.clients.audio=${audio_clients} \
        experiment.clients.image=${image_clients} \
        experiment.clients.audio_image=${both_clients}"
    
    if eval ${CMD}; then
        echo -e "${GREEN}✓ Experiment ${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS} completed successfully${NC}"
    else
        echo -e "${RED}✗ Experiment ${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS} failed${NC}"
        exit 1
    fi
    
    echo ""
}

START_TIME=$(date +%s)
echo -e "${GREEN}Starting all experiments at $(date)${NC}"
echo ""

for partitioner in iid niid; do
    for aggregation in full-model private-head private-head-fusion; do
        run_experiment ${partitioner} ${aggregation} 0 0 10
        run_experiment ${partitioner} ${aggregation} 3 3 4
        run_experiment ${partitioner} ${aggregation} 5 5 0
    done
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All experiments completed successfully!${NC}"
echo -e "${GREEN}Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s${NC}"
echo -e "${GREEN}Finished at $(date)${NC}"
echo -e "${GREEN}========================================${NC}"