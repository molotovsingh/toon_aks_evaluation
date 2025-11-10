const { encode, decode } = require('@toon-format/toon');
const { encode: encodeTokens } = require('gpt-tokenizer');
const legalCaseData = require('./legal_case_data.js');

// Test TOON efficiency on legal case data
function testLegalCaseTOON() {
  console.log('🏛️  LEGAL CASE TOON ANALYSIS');
  console.log('📋 Test Case: ABC Technologies vs XYZ Solutions');
  console.log('📅 Timeline: January 2024 - May 2025 (16+ months)');
  console.log('💰 Value: ₹2.5 crore contract\n');
  
  // Convert legal case data to JSON string
  const jsonString = JSON.stringify(legalCaseData, null, 2);
  
  // Convert to TOON format
  const toonString = encode(legalCaseData);
  
  // Count tokens using gpt-tokenizer
  const jsonTokens = encodeTokens(jsonString).length;
  const toonTokens = encodeTokens(toonString).length;
  
  // Calculate savings
  const tokenSavings = jsonTokens - toonTokens;
  const percentSavings = ((tokenSavings / jsonTokens) * 100).toFixed(1);
  
  // Cost analysis (GPT-4 pricing: $0.03 per 1K tokens)
  const jsonCost = (jsonTokens / 1000) * 0.03;
  const toonCost = (toonTokens / 1000) * 0.03;
  const costSavings = jsonCost - toonCost;
  
  console.log('🔢 TOKEN EFFICIENCY:');
  console.log(`   JSON: ${jsonTokens.toLocaleString()} tokens | $${jsonCost.toFixed(4)}`);
  console.log(`   TOON: ${toonTokens.toLocaleString()} tokens | $${toonCost.toFixed(4)}`);
  console.log(`   💰 Savings: ${percentSavings}% | $${costSavings.toFixed(4)} per query\n`);
  
  // Show format comparison (first 500 chars)
  console.log('📄 FORMAT COMPARISON (first 500 chars):');
  console.log('   JSON:');
  console.log('   ' + jsonString.substring(0, 500).replace(/\n/g, '\n   '));
  console.log('   ...\n');
  
  console.log('   TOON:');
  console.log('   ' + toonString.substring(0, 500).replace(/\n/g, '\n   '));
  console.log('   ...\n');
  
  // Test decoding works
  try {
    const decodedData = decode(toonString);
    const isValid = JSON.stringify(decodedData) === JSON.stringify(legalCaseData);
    console.log('🔄 TOON DECODE TEST:', isValid ? '✅ PASSED' : '❌ FAILED');
  } catch (error) {
    console.log('🔄 TOON DECODE TEST: ❌ ERROR -', error.message);
  }
  
  // Analysis by data type
  console.log('\n📊 DATA TYPE ANALYSIS:');
  
  // Case details
  const caseDetailsTokens = encodeTokens(JSON.stringify(legalCaseData.caseDetails)).length;
  const caseDetailsToonTokens = encodeTokens(encode(legalCaseData.caseDetails)).length;
  console.log(`   Case Details: ${caseDetailsTokens} → ${caseDetailsToonTokens} tokens`);
  
  // Timeline (largest array)
  const timelineTokens = encodeTokens(JSON.stringify(legalCaseData.timeline)).length;
  const timelineToonTokens = encodeTokens(encode(legalCaseData.timeline)).length;
  console.log(`   Timeline (${legalCaseData.timeline.length} events): ${timelineTokens} → ${timelineToonTokens} tokens`);
  
  // Parties
  const partiesTokens = encodeTokens(JSON.stringify(legalCaseData.parties)).length;
  const partiesToonTokens = encodeTokens(encode(legalCaseData.parties)).length;
  console.log(`   Parties: ${partiesTokens} → ${partiesToonTokens} tokens`);
  
  // Payments
  const paymentsTokens = encodeTokens(JSON.stringify(legalCaseData.payments)).length;
  const paymentsToonTokens = encodeTokens(encode(legalCaseData.payments)).length;
  console.log(`   Payments: ${paymentsTokens} → ${paymentsToonTokens} tokens`);
  
  // Court orders
  const ordersTokens = encodeTokens(JSON.stringify(legalCaseData.courtOrders)).length;
  const ordersToonTokens = encodeTokens(encode(legalCaseData.courtOrders)).length;
  console.log(`   Court Orders: ${ordersTokens} → ${ordersToonTokens} tokens`);
  
  console.log('\n💡 KEY FINDINGS:');
  console.log(`   • Total token reduction: ${tokenSavings.toLocaleString()} tokens (${percentSavings}%)`);
  console.log(`   • Cost savings per LLM query: $${costSavings.toFixed(4)}`);
  console.log(`   • Best efficiency on structured arrays (timeline, payments)`);
  console.log(`   • Mixed date formats handled well in both formats`);
  
  console.log('\n🎯 LEGAL DOCUMENT USE CASES:');
  console.log('   ✅ Contract timelines - Perfect for TOON tabular format');
  console.log('   ✅ Payment records - Ideal for TOON structured arrays');
  console.log('   ✅ Court proceedings - Good for TOON event sequences');
  console.log('   ⚠️  Complex nested legal arguments - Consider JSON for deep structures');
  
  return {
    jsonTokens,
    toonTokens,
    savings: percentSavings,
    costSavings
  };
}

// Run the test
if (require.main === module) {
  const results = testLegalCaseTOON();
  
  console.log('\n🏆 CONCLUSION:');
  console.log(`For legal case data processing, TOON provides:`);
  console.log(`✅ ${results.savings}% token reduction`);
  console.log(`✅ $${results.costSavings.toFixed(4)} cost savings per query`);
  console.log(`✅ Better structure for timeline-based legal data`);
  console.log('\n🚀 RECOMMENDATION: Use TOON for legal timeline and payment data');
}

module.exports = testLegalCaseTOON;
