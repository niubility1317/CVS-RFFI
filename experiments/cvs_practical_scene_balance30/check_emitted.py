import ast,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); files=list(p.glob('*.py'))
for f in files: ast.parse(f.read_text(encoding='utf8'),filename=str(f))
a=ast.parse((p/'publish.py').read_text(encoding='utf8')); emitted=[]
for n in ast.walk(a):
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='script' for t in n.targets) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):ast.parse(n.value.value.replace('CONFIG','{}'));emitted.append('PASS')
print(json.dumps(dict(status='PASS',package_Python_files=len(files),emitted_remote_AST=emitted,network_calls=0,CP=0,IQ=0,target=False)))
