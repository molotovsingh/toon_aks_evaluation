# TOON Evaluation Project

A clean, focused repository for evaluating TOON (Token-Oriented Object Notation) format efficiency and cost savings for LLM data processing.

## 🎯 Project Overview

This repository contains a comprehensive evaluation toolkit for measuring the real-world benefits of using TOON format instead of JSON when sending structured data to Large Language Models.

## 📁 Project Structure

```
toon_aks_evaluation/
├── README.md                    # This file
├── README-TOON-Analysis.md      # Complete TOON analysis results
├── INSIGHTS.md                  # Key findings and insights
├── toon-quick-demo.js          # Quick 30-second demonstration
├── toon-cost-analysis.js       # Comprehensive cost analysis tool
├── toon-quality-test.js        # Quality comparison testing
├── package.json                # Node.js dependencies
└── .git/                      # Version control
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
npm install
```

### 2. Run Quick Demo (30 seconds)
```bash
node toon-quick-demo.js
```

### 3. Full Cost Analysis (2 minutes)
```bash
node toon-cost-analysis.js
```

### 4. Quality Testing
```bash
node toon-quality-test.js
```

### 5. Interactive Analysis
```bash
node toon-cost-analysis.js --interactive
```

## 📊 Key Results Summary

Our evaluation across 5 different data types shows:

- **50% average token reduction** across all datasets
- **33% cost savings** with maintained or improved quality
- **6.1% quality improvement** with TOON format
- **Real cost impact**: $0.0019 - $0.1138 per query

See `README-TOON-Analysis.md` for complete results and analysis.

## 📖 Documentation

- **[README-TOON-Analysis.md](README-TOON-Analysis.md)** - Complete analysis results, cost comparisons, and integration guide
- **[INSIGHTS.md](INSIGHTS.md)** - Key findings and actionable recommendations

## 🔧 Dependencies

- `@toon-format/toon` - TOON encoding/decoding library
- `gpt-tokenizer` - Accurate token counting for cost calculations

## 🎯 What This Project Contains

✅ **TOON Cost Analysis Tools** - Measure token and dollar savings
✅ **Quality Comparison Testing** - Verify TOON maintains or improves output quality  
✅ **Interactive Analysis** - Test with your own data
✅ **Complete Documentation** - Integration guides and best practices
✅ **Real-world Examples** - Practical use cases and results

## 📁 Archived Content

All non-TOON related content from the original repository has been archived to `/archive/non_toon_files/` to maintain a clean, focused TOON evaluation environment.

## 💡 Next Steps

1. Review the complete analysis in `README-TOON-Analysis.md`
2. Run the tools with your own data using interactive mode
3. Integrate TOON into your LLM workflows for immediate cost savings
4. Monitor token usage and quality improvements

---

**Repository Status**: Clean TOON evaluation environment
**Last Updated**: 2025-11-09
**Version**: 1.0 (Post-cleanup)
