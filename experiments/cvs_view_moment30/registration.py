"""Expand registered path prefixes without altering method names or descriptions."""
def expand_paths(value,project):
 if isinstance(value,str):return project+value[len('PROJECT'):] if value.startswith('PROJECT/') else value
 if isinstance(value,list):return [expand_paths(v,project) for v in value]
 if isinstance(value,dict):return {k:expand_paths(v,project) for k,v in value.items()}
 return value
