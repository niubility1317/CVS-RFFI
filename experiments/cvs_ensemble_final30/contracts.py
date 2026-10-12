from experiments.cvs_phase1_stack.recover import verify_contract

def verified_component_contract(actual_contracts,reference):
 if len(actual_contracts)!=2:raise ValueError('Both actual new member contracts required')
 for actual in actual_contracts:verify_contract(actual,reference)
 # Preserve the runtime comparison established by the actual source loader.
 # A reference recipe is not runtime evidence; never synthesize its missing flag.
 return dict(actual_contracts[0])
