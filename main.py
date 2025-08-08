from typing import Callable

import jaclang
from jaclang import JacMachineInterface as Jac
from enum import Enum, auto
from mtllm import Model, Image, by
from dataclasses import dataclass


model = 'gemma3:12b-it-qat'
provider = 'ollama'
model_name = f'{provider}/{model}'
# model_name  = 'openai/gpt-4o-mini'
llm = Model(model_name=model_name, temperature=0, api_base="http://localhost:11434")


# llm = Model(model_name='gpt-4o-mini', temperature=0.1)

# @Jac.sem('', {'LEASE_AGREEMENT': '', 'LOAN_AGREEMENT': '', 'OTHER': ''})
class DocumentType(Enum):
    LEASE_AGREEMENT = 'Lease Agreement'
    LOAN_AGREEMENT = 'Loan Agreement'
    OTHER = 'Other'


@dataclass
class Address:
    address_line_1: str
    city: str
    state: str
    zip_code: int


@dataclass
class LoanAgreement:
    document_type: DocumentType
    lender_name: str
    lender_address: Address
    borrower_name: str
    borrower_address: Address
    principal_amount: float
    interest_rate: float
    start_date: str
    end_date: str


@dataclass
class LeaseAgreement:
    document_type: DocumentType
    bank_name: str
    bank_address: Address
    lessee_name: str
    lessee_address: Address
    vehicle_description: str
    vehicle_vin: str
    monthly_lease_payment: float
    security_deposit: float
    lease_start_date: str
    lease_end_date: str


@by(llm)
def classify_document_type(document: Image) -> DocumentType: ...


@by(llm)
def extract_lease_agreement_details(document: Image) -> LeaseAgreement: ...


@by(llm)
def extract_loan_agreement_details(document: Image) -> LoanAgreement: ...


document = Image('image.png')

# doc_type = classify_document_type(document)

try:
    doc_type = classify_document_type(document)
except Exception as e:
    print(f"Error classifying document type: {e}")
    exit(1)

print('Document Type: ', doc_type)
if doc_type == DocumentType.LEASE_AGREEMENT:
    lease_details = extract_lease_agreement_details(document)
    print('Lease Agreement Details: \n', lease_details)
elif doc_type == DocumentType.LOAN_AGREEMENT:
    loan_details = extract_loan_agreement_details(document)
    print('Loan Agreement Details: \n', loan_details)
else:
    print('Document type is not recognized or is other.')
