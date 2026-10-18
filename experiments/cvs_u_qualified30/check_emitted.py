"""Parse the package and emitted remote payload without executing any remote action."""
import ast,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); files=list(p.glob('*.py'))
for f in files: ast.parse(f.read_text(encoding='utf8'),filename=str(f))
a=ast.parse((p/'publish.py').read_text(encoding='utf8')); emitted=[]
for n in ast.walk(a):
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='script' for t in n.targets) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):ast.parse(n.value.value.replace('CONFIG','{}'));emitted.append('PASS')
assert emitted,'The remote release script must be parsed'
result=dict(status='PASS',package_Python_files=len(files),emitted_remote_AST=emitted,network_calls=0,CP=0,IQ=0,NN=0,target=False)
if len(sys.argv)>2:
 out=pathlib.Path(sys.argv[2]);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result))
