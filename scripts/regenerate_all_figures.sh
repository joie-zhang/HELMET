#!/bin/bash
# Master script to regenerate all figures for the paper
# This streamlines the figure revision process

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Figure Regeneration Script"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Collect/update results data
echo -e "${BLUE}Step 1: Collecting results data...${NC}"
python scripts/collect_results_new.py
echo -e "${GREEN}✓ Results collected${NC}"
echo ""

# Step 2: Generate main comparison plots
echo -e "${BLUE}Step 2: Generating main comparison plots...${NC}"
python scripts/plot_averages_comparison.py
echo -e "${GREEN}✓ Main comparison plots generated${NC}"
echo ""

# Step 3: Generate HELMET-specific plots
echo -e "${BLUE}Step 3: Generating HELMET plots...${NC}"
python scripts/plot_results_helmet.py
echo -e "${GREEN}✓ HELMET plots generated${NC}"
echo ""

# Step 4: Generate LongProc-specific plots
echo -e "${BLUE}Step 4: Generating LongProc plots...${NC}"
python scripts/plot_results_longproc.py
echo -e "${GREEN}✓ LongProc plots generated${NC}"
echo ""

# Step 5: Generate quadrant comparison plots (if needed)
if [ -f "scripts/plot_quadrant_comparison.py" ]; then
    echo -e "${BLUE}Step 5: Generating quadrant comparison plots...${NC}"
    python scripts/plot_quadrant_comparison.py
    echo -e "${GREEN}✓ Quadrant plots generated${NC}"
    echo ""
fi

# Step 6: Generate task delta plots (if needed)
if [ -f "scripts/plot_task_deltas_averaged_configs.py" ]; then
    echo -e "${BLUE}Step 6: Generating task delta plots...${NC}"
    python scripts/plot_task_deltas_averaged_configs.py
    python scripts/plot_task_deltas_separate_configs.py
    echo -e "${GREEN}✓ Task delta plots generated${NC}"
    echo ""
fi

# Step 7: Generate ICL-specific plots (if needed)
if [ -f "scripts/plot_icl_memory_only.py" ]; then
    echo -e "${BLUE}Step 7: Generating ICL memory plots...${NC}"
    python scripts/plot_icl_memory_only.py
    echo -e "${GREEN}✓ ICL plots generated${NC}"
    echo ""
fi

echo -e "${GREEN}=========================================="
echo "All figures regenerated successfully!"
echo "==========================================${NC}"
echo ""
echo "Output directory: results/plots/"
echo ""
echo "Key figures:"
echo "  - averages_comparison_1x1_all_techniques_with_connections_incl_duo.png/pdf"
echo "  - helmet_overall_plot.png"
echo "  - longproc_overall_plot.png"
echo ""

