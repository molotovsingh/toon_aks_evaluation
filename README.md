# TOON vs JSON Cost Analysis

A practical tool to measure the real-world cost savings of using TOON (Token-Oriented Object Notation) instead of JSON for LLM data input.

## 🎯 What This Demonstrates

This analysis shows the **actual token and dollar cost differences** between JSON and TOON formats when sending data to Large Language Models. The results confirm TOON's claims of 30-60% token reduction with real, measurable cost savings.

## 📊 Key Results Summary

From our comprehensive analysis across 5 different data types:

| Dataset | Token Savings | Cost Savings (GPT-4) | Best Use Case |
|---------|---------------|---------------------|---------------|
| **Employee Records** | 63.1% | $0.11 per query | Perfect tabular data |
| **Analytics Data** | 61.4% | $0.08 per query | Time-series metrics |
| **E-commerce Orders** | 38.2% | $0.10 per query | Nested structures |
| **Mixed Data** | 22.4% | $0.002 per query | Semi-uniform arrays |
| **Small Datasets** | 21.0% | $0.003 per query | Simple metadata |

**Total Savings: 50.1% tokens across all datasets**

## 🚀 Quick Start

### 1. Quick Demo (30 seconds)
```bash
node toon-quick-demo.js
```
Shows a simple 3-employee example with immediate cost comparison.

### 2. Full Analysis (2 minutes)
```bash
node toon-cost-analysis.js
```
Comprehensive analysis across 5 different data types with detailed cost breakdowns.

### 3. Interactive Testing
```bash
node toon-cost-analysis.js --interactive
```
Enter your own JSON data to see personalized cost savings.

## 💰 Real Cost Impact

### Per Query Savings (GPT-4 pricing):
- **Small dataset (3 employees)**: $0.0019 savings
- **Medium dataset (100 employees)**: $0.1138 savings  
- **Large dataset (60 analytics records)**: $0.0788 savings

### At Scale (1,000 queries):
- **Total cost savings**: $300.72 with GPT-4
- **Total cost savings**: $150.36 with Claude-3
- **Total cost savings**: $15.04 with GPT-3.5-turbo

### At Scale (10,000 queries):
- **Total cost savings**: $3,007.20 with GPT-4
- **Total cost savings**: $1,503.60 with Claude-3  
- **Total cost savings**: $150.36 with GPT-3.5-turbo

## 📈 Format Comparison

**JSON (108 tokens, $0.0032):**
```json
{
  "employees": [
    {"id": 1, "name": "Alice", "role": "Engineer", "salary": 120000},
    {"id": 2, "name": "Bob", "role": "Designer", "salary": 95000},
    {"id": 3, "name": "Charlie", "role": "Manager", "salary": 140000}
  ]
}
```

**TOON (44 tokens, $0.0013):**
```
employees[3]{id,name,role,salary}:
  1,Alice,Engineer,120000
  2,Bob,Designer,95000
  3,Charlie,Manager,140000
```

**Result: 59.3% token reduction, $0.0019 savings per query**

## 🎯 When to Use TOON

### ✅ Best for TOON:
- **Employee/user records** - Perfect tabular data
- **Analytics/metrics data** - Time-series with consistent fields
- **Product catalogs** - Uniform product information
- **Transaction logs** - Repeated transaction structures
- **Any uniform array of objects** with the same fields

### ⚠️ Consider JSON for:
- **Highly nested data** - Deep object hierarchies
- **Variable schemas** - Objects with very different fields
- **Small datasets** - Under 10 items (overhead not worth it)
- **Complex relationships** - Deeply interconnected data

## 🔧 Integration Guide

### Step 1: Convert JSON to TOON before LLM calls
```javascript
const { encode } = require('@toon-format/toon');

// Your existing code
const userData = getUserData(); // Returns JSON object

// Convert to TOON for LLM efficiency
const toonData = encode(userData);

// Send to LLM
const response = await llm.generate({
  prompt: `Analyze this user data: ${toonData}`
});
```

### Step 2: Decode TOON responses back to JSON
```javascript
const { decode } = require('@toon-format/toon');

const toonResponse = await llm.generate(prompt);
const jsonData = decode(toonResponse);
```

### Step 3: Calculate your actual savings
```javascript
const { encode: encodeTokens } = require('gpt-tokenizer');

const jsonSize = encodeTokens(JSON.stringify(data)).length;
const toonSize = encodeTokens(encode(data)).length;
const savings = ((jsonSize - toonSize) / jsonSize * 100).toFixed(1);

console.log(`TOON saves ${savings}% tokens for your data`);
```

## 📋 Data Requirements for TOON

TOON works best when your data meets these criteria:

1. **Uniform Objects**: All objects in an array have the same fields
2. **Primitive Values**: Fields contain strings, numbers, booleans, or null
3. **Consistent Structure**: No missing fields or varying field types
4. **Tabular Nature**: Data looks like a table/spreadsheet

Example of ideal TOON data:
```javascript
{
  employees: [
    { id: 1, name: "Alice", department: "Engineering", salary: 120000 },
    { id: 2, name: "Bob", department: "Design", salary: 95000 },
    { id: 3, name: "Charlie", department: "Marketing", salary: 100000 }
  ]
}
```

## 🧮 How the Analysis Works

1. **Sample Data Generation**: Creates realistic datasets for different use cases
2. **Format Conversion**: Converts each dataset to both JSON and TOON
3. **Token Counting**: Uses GPT tokenizer to count actual tokens
4. **Cost Calculation**: Applies current LLM pricing rates
5. **Comparison Report**: Generates detailed before/after analysis

## 📦 Dependencies

- `@toon-format/toon` - TOON encoding/decoding
- `gpt-tokenizer` - Accurate token counting

## 🔍 Understanding the Results

### Token Efficiency
- **60%+ savings**: Employee records, analytics data (perfect tabular)
- **30-40% savings**: E-commerce orders (nested but mostly uniform)  
- **20-30% savings**: Mixed data (some uniform, some variable)
- **<20% savings**: Complex nested data (TOON may not be optimal)

### Cost Impact
- **High-cost LLMs** (GPT-4, Claude-3): Significant savings
- **Medium-cost LLMs** (GPT-3.5): Moderate savings
- **Low-cost LLMs** (Haiku): Small but measurable savings

### Scale Factor
- **Cost savings compound** with the number of queries
- **ROI increases** with dataset size and LLM cost
- **Break-even** is essentially instant for any meaningful usage

## 💡 Pro Tips

1. **Test Your Data**: Run the interactive mode with your actual datasets
2. **Batch Processing**: Apply TOON to your largest, most frequent queries first
3. **Monitor Savings**: Track token usage before/after TOON implementation
4. **Use Tab Delimiters**: For very large datasets, try `--delimiter "\t"` option
5. **Keep JSON for Storage**: Use TOON only for LLM input, JSON for databases

## 🏆 Bottom Line

TOON delivers on its promises:
- **Measured 50.1% token reduction** across diverse datasets
- **Real cost savings** starting from the very first query
- **Scales dramatically** with usage volume
- **Simple integration** into existing workflows

**If you send structured data to LLMs and pay for tokens, TOON will save you money.**

## 📖 Learn More

- [TOON Specification](https://github.com/toon-format/spec)
- [Official Website](https://toonformat.dev)
- [Interactive Playground](https://www.curiouslychase.com/playground/format-tokenization-exploration)

---

*This analysis was generated on 2025-11-09 using real token counting and current LLM pricing rates.*
