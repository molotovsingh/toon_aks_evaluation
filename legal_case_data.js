// ABC Technologies vs XYZ Solutions - Legal Case Data
// Extracted from PDF with mixed date formats

const legalCaseData = {
  caseDetails: {
    caseId: "ABC-XYZ-2024-001",
    title: "ABC Technologies vs XYZ Solutions - Contract Dispute",
    court: "Delhi High Court",
    contractValue: 25000000,
    currency: "INR"
  },
  parties: [
    {
      id: "P001",
      name: "ABC Technologies Pvt. Ltd.",
      type: "plaintiff",
      role: "client"
    },
    {
      id: "P002", 
      name: "XYZ Solutions LLP",
      type: "defendant",
      role: "vendor"
    }
  ],
  timeline: [
    {
      date: "2024-01-01",
      event: "Contract signed",
      description: "Detailed software development contract for enterprise suite",
      amount: null,
      parties: ["P001", "P002"]
    },
    {
      date: "2024-01-10",
      event: "Initial advance payment",
      description: "ABC transferred initial advance",
      amount: 10000000,
      parties: ["P001"]
    },
    {
      date: "2024-01-25", 
      event: "First joint review",
      description: "Project teams discussed technical customization at ABC's office",
      amount: null,
      parties: ["P001", "P002"]
    },
    {
      date: "2024-02-15",
      event: "Working roadmap presentation",
      description: "XYZ presented working roadmap during virtual meeting",
      amount: null,
      parties: ["P002", "P001"]
    },
    {
      date: "2024-02-28",
      event: "Additional modules request",
      description: "ABC requested additional modules",
      amount: null,
      parties: ["P001"]
    },
    {
      date: "2024-03-05",
      event: "Milestone report",
      description: "XYZ circulated milestone report",
      amount: null,
      parties: ["P002"]
    },
    {
      date: "2024-03-08",
      event: "Concerns raised",
      description: "ABC emailed concerns over missed functional targets",
      amount: null,
      parties: ["P001"]
    },
    {
      date: "2024-03-18",
      event: "Further payment",
      description: "ABC paid further amount contingent on improved timelines",
      amount: 7500000,
      parties: ["P001"]
    },
    {
      date: "2024-04-01",
      event: "Weekly update period start",
      description: "Weekly updates began highlighting backend issues",
      amount: null,
      parties: ["P002"]
    },
    {
      date: "2024-05-21",
      event: "CTO site visit",
      description: "ABC's CTO visited XYZ's premises for progress demo, uncovered database issues",
      amount: null,
      parties: ["P001", "P002"]
    },
    {
      date: "2024-06-05",
      event: "Written warning",
      description: "ABC issued written warning regarding milestone defaults, referencing losses",
      amount: null,
      parties: ["P001"]
    },
    {
      date: "2024-06-22",
      event: "Extension request",
      description: "XYZ requested extension until 31.08.2024",
      amount: null,
      parties: ["P002"]
    },
    {
      date: "2024-06-30",
      event: "Extension rejected",
      description: "ABC rejected extension request due to growing losses",
      amount: null,
      parties: ["P001"]
    },
    {
      date: "2024-07-05",
      event: "Negotiation period start",
      description: "Several negotiation meetings began (July 5-20)",
      amount: null,
      parties: ["P001", "P002"]
    },
    {
      date: "2024-07-20",
      event: "Negotiations end",
      description: "Negotiations ended inconclusively",
      amount: null,
      parties: ["P001", "P002"]
    },
    {
      date: "2024-08-02",
      event: "Breach notice served",
      description: "ABC served breach notice demanding completion or ₹1.75 crore refund",
      amount: 17500000,
      parties: ["P001"]
    },
    {
      date: "2024-08-10",
      event: "Defendant response",
      description: "XYZ proposed to refund only ₹50 lakh and push for revised timelines",
      amount: 5000000,
      parties: ["P002"]
    },
    {
      date: "2024-08-18",
      event: "Commercial Suit filed",
      description: "ABC filed Commercial Suit seeking specific performance, damages ₹1 crore, interim injunction",
      amount: 10000000,
      parties: ["P001"]
    },
    {
      date: "2024-09-10",
      event: "Written Statement",
      description: "Defendant filed Written Statement arguing force majeure under Section 56 Contract Act",
      amount: null,
      parties: ["P002"]
    },
    {
      date: "2024-09-15",
      event: "Dismissal application",
      description: "XYZ filed dismissal application under Order VII Rule 11 CPC",
      amount: null,
      parties: ["P002"]
    },
    {
      date: "2024-11-22",
      event: "Oral arguments commence",
      description: "High Court hearing with ABC relying on contract terms, XYZ on doctrine of frustration",
      amount: null,
      parties: ["P001", "P002"]
    },
    {
      date: "2024-12-20",
      event: "High Court judgment",
      description: "Court directed XYZ to refund ₹1 crore, pay damages ₹50 lakh, 2-year restraint",
      amount: 15000000,
      parties: ["P001", "P002"]
    },
    {
      date: "2025-01-15",
      event: "Appeal filed",
      description: "XYZ appealed damages figure",
      amount: null,
      parties: ["P002"]
    },
    {
      date: "2025-05-15",
      event: "Appellate court ruling",
      description: "Court affirmed liability but reduced damages to ₹30 lakh, sent for mediation",
      amount: 3000000,
      parties: ["P001", "P002"]
    }
  ],
  payments: [
    {
      date: "2024-01-10",
      amount: 10000000,
      type: "advance",
      from: "P001",
      to: "P002"
    },
    {
      date: "2024-03-18", 
      amount: 7500000,
      type: "contingent",
      from: "P001",
      to: "P002"
    }
  ],
  courtOrders: [
    {
      date: "2024-12-20",
      type: "judgment",
      amount: 15000000,
      description: "Refund ₹1 crore + damages ₹50 lakh + 2-year restraint"
    },
    {
      date: "2025-05-15",
      type: "appellate_ruling", 
      amount: 3000000,
      description: "Affirmed liability, reduced damages to ₹30 lakh, mediation ordered"
    }
  ]
};

module.exports = legalCaseData;
