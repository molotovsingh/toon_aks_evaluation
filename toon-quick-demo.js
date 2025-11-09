#!/usr/bin/env node

const { encode } = require('@toon-format/toon');
const { encode: encodeTokens } = require('gpt-tokenizer');

/**
 * Quick TOON Demo - Simple interactive test
 */

function demoQuick() {
  console.log('🚀 TOON vs JSON Quick Demo\n');
  
  // Simple employee data
  const employeeData = {
    employees: [
      { id: 1, name: 'Alice', role: 'Engineer', salary: 120000 },
      { id: 2, name: 'Bob', role: 'Designer', salary: 95000 },
      { id: 3, name: 'Charlie', role: 'Manager', salary: 140000 }
    ]
  };
  
  const jsonString = JSON.stringify(employeeData, null, 2);
  const toonString = encode(employeeData);
  
  const jsonTokens = encodeTokens(jsonString).length;
  const toonTokens = encodeTokens(toonString).length;
  const savings = ((jsonTokens - toonTokens) / jsonTokens * 100).toFixed(1);
  
  console.log('📊 Sample Data: 3 Employees');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  
  console.log('\n💰 COST COMPARISON:');
  console.log(`   JSON: ${jsonTokens} tokens | $${((jsonTokens / 1000) * 0.03).toFixed(4)} (GPT-4)`);
  console.log(`   TOON: ${toonTokens} tokens | $${((toonTokens / 1000) * 0.03).toFixed(4)} (GPT-4)`);
  console.log(`   🎯 SAVINGS: ${savings}% | $${(((jsonTokens - toonTokens) / 1000) * 0.03).toFixed(4)} per query`);
  
  console.log('\n📄 JSON Format:');
  console.log('   ' + jsonString.replace(/\n/g, '\n   '));
  
  console.log('\n🎒 TOON Format:');
  console.log('   ' + toonString.replace(/\n/g, '\n   '));
  
  console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('💡 For larger datasets, savings scale dramatically!');
  console.log('   Run: node toon-cost-analysis.js  (for full analysis)');
  console.log('   Run: node toon-cost-analysis.js --interactive  (enter your own data)');
}

// Run demo
demoQuick();
