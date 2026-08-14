# compliance_fixtures.py

"""
These are static but later can be connected to a real regulatory knowledge base (vector DB) for live queries.
"""
from src.compliance_kb import CompliancePassage

REG_B_FCRA_FIXTURES = [
    # --- REGULATION B (ECOA) SEED DATA ---
    CompliancePassage(
        id="reg_b_adverse_action_notice",
        content="A creditor shall notify an applicant of action taken within 30 days after receiving a completed application concerning the creditor's approval of, counteroffer to, or adverse action on the application.",
        citation="12 CFR § 1002.9(a)(1)",
        metadata={
            "statute": "Reg B", 
            "topic": "Adverse Action Timing",
            "source_url": "https://www.ecfr.gov/current/title-12/chapter-X/part-1002/subpart-A/section-1002.9",
            "governing_body": "CFPB"
        }
    ),
    CompliancePassage(
        id="reg_b_prohibited_factors",
        content="It shall be unlawful for any creditor to discriminate against any applicant, with respect to any aspect of a credit transaction, on the basis of race, color, religion, national origin, sex or marital status, or age (provided the applicant has the capacity to contract).",
        citation="12 CFR § 1002.4(a)",
        metadata={
            "statute": "Reg B", 
            "topic": "Prohibited Discrimination",
            "source_url": "https://www.ecfr.gov/current/title-12/chapter-X/part-1002/subpart-A/section-1002.4",
            "governing_body": "CFPB"
        }
    ),
    CompliancePassage(
        id="reg_b_specific_reasons",
        content="A statement of reasons for adverse action required by this section must be specific and indicate the principal reason(s) for the adverse action. Statements that the adverse action was based on the creditor's internal standards or policies or that the applicant failed to achieve a qualifying score on the creditor's credit scoring system are insufficient.",
        citation="12 CFR § 1002.9(b)(2)",
        metadata={
            "statute": "Reg B",
            "topic": "Specificity of Adverse Action",
            "source_url": "https://www.ecfr.gov/current/title-12/chapter-X/part-1002/subpart-A/section-1002.9#p-1002.9(b)(2)",
            "governing_body": "CFPB"
        }
    ),

    # --- FCRA (FAIR CREDIT REPORTING ACT) SEED DATA ---
    CompliancePassage(
        id="fcra_adverse_action_disclosure",
        content="If any person takes any adverse action with respect to any consumer that is based in whole or in part on any information contained in a consumer report, the person shall provide notice of the adverse action to the consumer, include the name, address, and telephone number of the consumer reporting agency, and a statement that the consumer reporting agency did not make the decision to take the adverse action.",
        citation="15 U.S.C. § 1681m(a)",
        metadata={
            "statute": "FCRA", 
            "topic": "Consumer Report Disclosure",
            "source_url": "https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title15-section1681m&num=0&edition=prelim",
            "governing_body": "FTC / CFPB"
        }
    )
]