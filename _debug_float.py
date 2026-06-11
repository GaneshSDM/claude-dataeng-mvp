"""Debug float_equality rule"""
import sqlglot
from sqlglot import exp

tree = sqlglot.parse_one("SELECT * FROM orders WHERE unit_price = 19.99")
print("Tree:", repr(tree.sql()))
for eq in tree.find_all(exp.EQ):
    print("EQ:", eq)
    for literal in eq.find_all(exp.Literal):
        print("  Literal found")
        print("  is_string:", literal.args.get("is_string"))
        print("  this:", repr(literal.args.get("this")), type(literal.args.get("this")))
        val = literal.args.get("this", "")
        print("  str(this):", repr(str(val)))
        is_str = literal.args.get("is_string")
        print("  is_string is False:", is_str is False)
        print("  '.' in str(val):", "." in str(val))

# Test with our actual rule logic
findings = []
for eq in tree.find_all(exp.EQ):
    for literal in eq.find_all(exp.Literal):
        if literal.args.get("is_string") is False and "." in str(literal.args.get("this", "")):
            findings.append({"found": True})
            break
print("Findings:", findings)

# Also test: what does _float_equality return?
from core.sql_engine.rules import _float_equality
print("Rule output:", _float_equality(tree))