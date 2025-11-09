#!/usr/bin/env node

const { encode, decode } = require('@toon-format/toon');
const { encode: encodeTokens } = require('gpt-tokenizer');

/**
 * TOON vs JSON Cost Analysis Tool
 * Compares token usage and cost between JSON and TOON formats
 */

// LLM Pricing (per 1K tokens) - approximate current rates
const LLM_PRICING = {
  'gpt-4': { input: 0.03, output: 0.06 },
  'gpt-3.5-turbo': { input: 0.0015, output: 0.002 },
  'claude-3': { input: 0.015, output: 0.075 },
  'claude-3-haiku': { input: 0.00025, output: 0.00125 },
  'gemini-pro': { input: 0.00125, output: 0.0035 }
};

// Sample datasets representing different use cases
const SAMPLE_DATASETS = {
  employeeRecords: {
    name: 'Employee Records',
    description: 'Uniform employee data - optimal for TOON',
    data: {
      employees: Array.from({ length: 100 }, (_, i) => ({
        id: i + 1,
        name: `Employee ${i + 1}`,
        email: `employee${i + 1}@company.com`,
        department: ['Engineering', 'Sales', 'Marketing', 'HR'][i % 4],
        salary: 50000 + (i * 1000),
        yearsExperience: Math.floor(Math.random() * 15) + 1,
        active: i % 10 !== 0
      }))
    }
  },
  
  ecommerceOrders: {
    name: 'E-commerce Orders',
    description: 'Nested order data with customer info',
    data: {
      orders: Array.from({ length: 50 }, (_, i) => ({
        orderId: `ORD-${String(i + 1).padStart(4, '0')}`,
        customer: {
          id: Math.floor(Math.random() * 1000) + 1,
          name: `Customer ${i + 1}`,
          email: `customer${i + 1}@email.com`
        },
        items: Array.from({ length: Math.floor(Math.random() * 5) + 1 }, (_, j) => ({
          productId: `PROD-${String(j + 1).padStart(3, '0')}`,
          quantity: Math.floor(Math.random() * 3) + 1,
          price: (Math.random() * 100 + 10).toFixed(2)
        })),
        total: (Math.random() * 500 + 50).toFixed(2),
        status: ['pending', 'processing', 'shipped', 'delivered'][i % 4],
        createdAt: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString()
      }))
    }
  },

  analyticsData: {
    name: 'Time-series Analytics',
    description: 'Time-series metrics data',
    data: {
      metrics: Array.from({ length: 60 }, (_, i) => ({
        date: new Date(Date.now() - (59 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        pageViews: Math.floor(Math.random() * 10000) + 1000,
        uniqueVisitors: Math.floor(Math.random() * 8000) + 800,
        bounceRate: (Math.random() * 0.5 + 0.1).toFixed(3),
        avgSessionDuration: Math.floor(Math.random() * 300) + 60,
        conversions: Math.floor(Math.random() * 50) + 5,
        revenue: (Math.random() * 1000 + 100).toFixed(2)
      }))
    }
  },

  githubRepos: {
    name: 'GitHub Repositories',
    description: 'Repository metadata from popular projects',
    data: {
      repositories: [
        {
          id: 1,
          name: 'awesome-python',
          fullName: 'vinta/awesome-python',
          description: 'A curated list of awesome Python frameworks, libraries, software and resources',
          stars: 185000,
          forks: 18000,
          language: 'Python',
          createdAt: '2014-06-27T21:00:00Z',
          updatedAt: '2025-11-09T10:30:00Z',
          topics: ['python', 'awesome', 'lists', 'learning']
        },
        {
          id: 2,
          name: 'react',
          fullName: 'facebook/react',
          description: 'The library for web and native user interfaces',
          stars: 225000,
          forks: 45000,
          language: 'JavaScript',
          createdAt: '2013-05-24T16:15:54Z',
          updatedAt: '2025-11-09T09:15:00Z',
          topics: ['javascript', 'ui', 'library', 'facebook']
        },
        {
          id: 3,
          name: 'tensorflow',
          fullName: 'tensorflow/tensorflow',
          description: 'An Open Source Machine Learning Framework for Everyone',
          stars: 185000,
          forks: 74000,
          language: 'C++',
          createdAt: '2015-11-07T01:47:58Z',
          updatedAt: '2025-11-09T08:45:00Z',
          topics: ['machine-learning', 'tensorflow', 'deep-learning', 'neural-network']
        }
      ]
    }
  },

  mixedStructure: {
    name: 'Mixed Structure Data',
    description: 'Semi-uniform data with varying fields',
    data: {
      events: [
        {
          id: 1,
          type: 'user_login',
          timestamp: '2025-11-09T10:30:00Z',
          userId: 12345,
          details: { browser: 'Chrome', os: 'Windows', ip: '192.168.1.1' }
        },
        {
          id: 2,
          type: 'page_view',
          timestamp: '2025-11-09T10:31:00Z',
          userId: 12345,
          page: '/dashboard',
          referrer: 'google.com'
        },
        {
          id: 3,
          type: 'purchase',
          timestamp: '2025-11-09T10:32:00Z',
          userId: 12345,
          amount: 99.99,
          currency: 'USD',
          items: ['item1', 'item2']
        }
      ]
    }
  }
};

/**
 * Count tokens in text using GPT tokenizer
 */
function countTokens(text) {
  return encodeTokens(text).length;
}

/**
 * Format number with thousands separator
 */
function formatNumber(num) {
  return num.toLocaleString();
}

/**
 * Format currency
 */
function formatCurrency(amount) {
  return `$${amount.toFixed(4)}`;
}

/**
 * Analyze a single dataset
 */
function analyzeDataset(dataset) {
  const jsonString = JSON.stringify(dataset.data, null, 2);
  const toonString = encode(dataset.data);
  
  const jsonTokens = countTokens(jsonString);
  const toonTokens = countTokens(toonString);
  
  const tokenSavings = jsonTokens - toonTokens;
  const percentSavings = ((tokenSavings / jsonTokens) * 100).toFixed(1);
  
  const results = {
    name: dataset.name,
    description: dataset.description,
    json: {
      content: jsonString,
      tokens: jsonTokens,
      size: Buffer.byteLength(jsonString, 'utf8')
    },
    toon: {
      content: toonString,
      tokens: toonTokens,
      size: Buffer.byteLength(toonString, 'utf8')
    },
    savings: {
      tokens: tokenSavings,
      percent: parseFloat(percentSavings),
      size: Buffer.byteLength(jsonString, 'utf8') - Buffer.byteLength(toonString, 'utf8')
    }
  };
  
  // Calculate cost estimates for different LLMs
  results.costs = {};
  for (const [provider, pricing] of Object.entries(LLM_PRICING)) {
    const jsonCost = (jsonTokens / 1000) * pricing.input;
    const toonCost = (toonTokens / 1000) * pricing.input;
    const costSavings = jsonCost - toonCost;
    
    results.costs[provider] = {
      json: jsonCost,
      toon: toonCost,
      savings: costSavings,
      savingsPercent: ((costSavings / jsonCost) * 100).toFixed(1)
    };
  }
  
  return results;
}

/**
 * Print formatted analysis results
 */
function printResults(results) {
  console.log(`\n${'='.repeat(80)}`);
  console.log(`📊 DATASET: ${results.name}`);
  console.log(`${'='.repeat(80)}`);
  console.log(`📝 ${results.description}`);
  
  console.log(`\n🔢 TOKEN COMPARISON:`);
  console.log(`   JSON:  ${formatNumber(results.json.tokens)} tokens | ${formatNumber(results.json.size)} bytes`);
  console.log(`   TOON:  ${formatNumber(results.toon.tokens)} tokens | ${formatNumber(results.toon.size)} bytes`);
  console.log(`   💰 SAVINGS: ${formatNumber(results.savings.tokens)} tokens (${results.savings.percent}%) | ${formatNumber(results.savings.size)} bytes`);
  
  console.log(`\n💵 COST ESTIMATES (per 1K tokens):`);
  for (const [provider, cost] of Object.entries(results.costs)) {
    console.log(`   ${provider.padEnd(20)} | JSON: ${formatCurrency(cost.json)} | TOON: ${formatCurrency(cost.toon)} | SAVINGS: ${formatCurrency(cost.savings)} (${cost.savingsPercent}%)`);
  }
  
  // Show sample comparison
  console.log(`\n📄 FORMAT COMPARISON (first 300 chars):`);
  console.log(`   JSON:`);
  console.log(`   ${JSON.stringify(results.json.content).slice(0, 300)}...`);
  console.log(`   \n   TOON:`);
  console.log(`   ${results.toon.content.split('\n').slice(0, 5).join('\n   ')}...`);
  
  console.log(`${'='.repeat(80)}\n`);
}

/**
 * Generate summary report
 */
function generateSummary(allResults) {
  const totalJsonTokens = allResults.reduce((sum, r) => sum + r.json.tokens, 0);
  const totalToonTokens = allResults.reduce((sum, r) => sum + r.toon.tokens, 0);
  const totalTokenSavings = totalJsonTokens - totalToonTokens;
  const totalPercentSavings = ((totalTokenSavings / totalJsonTokens) * 100).toFixed(1);
  
  console.log(`\n${'='.repeat(80)}`);
  console.log(`🎯 SUMMARY REPORT`);
  console.log(`${'='.repeat(80)}`);
  console.log(`📈 TOTAL ACROSS ALL DATASETS:`);
  console.log(`   JSON Total: ${formatNumber(totalJsonTokens)} tokens`);
  console.log(`   TOON Total: ${formatNumber(totalToonTokens)} tokens`);
  console.log(`   Total Savings: ${formatNumber(totalTokenSavings)} tokens (${totalPercentSavings}%)`);
  
  console.log(`\n💡 COST IMPACT ANALYSIS:`);
  for (const [provider, pricing] of Object.entries(LLM_PRICING)) {
    const jsonTotalCost = (totalJsonTokens / 1000) * pricing.input;
    const toonTotalCost = (totalToonTokens / 1000) * pricing.input;
    const totalCostSavings = jsonTotalCost - toonTotalCost;
    
    // Calculate for 1000 queries
    const queries = 1000;
    const querySavings = totalCostSavings * queries;
    
    console.log(`   ${provider.padEnd(20)} | Per Query: ${formatCurrency(totalCostSavings)} | 1K Queries: ${formatCurrency(querySavings)} | 10K Queries: ${formatCurrency(querySavings * 10)}`);
  }
  
  console.log(`\n🚀 KEY INSIGHTS:`);
  console.log(`   • TOON provides significant token savings for uniform data structures`);
  console.log(`   • Greatest savings achieved with employee records (tabular data)`);
  console.log(`   • Mixed structures show moderate improvements`);
  console.log(`   • Cost savings compound with scale (1000+ queries)`);
  console.log(`   • ROI increases with higher-cost LLM providers`);
  
  console.log(`\n📋 RECOMMENDATIONS:`);
  console.log(`   ✅ Use TOON for: Employee records, analytics data, uniform arrays`);
  console.log(`   ⚠️  Consider JSON for: Deeply nested, highly variable structures`);
  console.log(`   💡 Pro Tip: Convert to TOON before sending to LLMs, use JSON for storage/processing`);
  
  console.log(`${'='.repeat(80)}\n`);
}

/**
 * Main execution
 */
function main() {
  console.log(`🔍 TOON vs JSON Cost Analysis Tool`);
  console.log(`📊 Comparing token usage and cost across different data structures\n`);
  
  // Analyze each dataset
  const results = [];
  for (const [key, dataset] of Object.entries(SAMPLE_DATASETS)) {
    const result = analyzeDataset(dataset);
    results.push(result);
    printResults(result);
  }
  
  // Generate summary
  generateSummary(results);
  
  // Interactive mode
  if (process.argv.includes('--interactive') || process.argv.includes('-i')) {
    console.log(`🔧 Interactive Mode - Enter your own JSON data:`);
    console.log(`   Paste JSON data and press Enter twice to analyze`);
    console.log(`   Type 'quit' to exit\n`);
    
    const readline = require('readline');
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });
    
    let inputData = '';
    let lineCount = 0;
    
    rl.on('line', (line) => {
      if (line.trim().toLowerCase() === 'quit') {
        rl.close();
        return;
      }
      
      if (line.trim() === '' && lineCount > 0) {
        // Empty line - analyze the data
        try {
          const customData = JSON.parse(inputData);
          const customDataset = {
            name: 'Custom Dataset',
            description: 'User-provided data',
            data: customData
          };
          const result = analyzeDataset(customDataset);
          printResults(result);
        } catch (error) {
          console.log(`❌ Error parsing JSON: ${error.message}`);
        }
        inputData = '';
        lineCount = 0;
      } else {
        inputData += line + '\n';
        lineCount++;
      }
    });
    
    rl.on('close', () => {
      console.log(`\n👋 Thanks for using TOON Cost Analysis!`);
    });
  }
}

// Run the analysis
if (require.main === module) {
  main();
}

module.exports = { analyzeDataset, countTokens, SAMPLE_DATASETS };
