from pyjexl.analysis import JEXLAnalyzer
from pyjexl.jexl import JEXL
from pyjexl import parser


def classify_filter(filter_expr):
    jexl = JEXL()
    classifications = jexl.analyze(filter_expr, FilterExpressionAnalyzer)
    if "unknown" in classifications:
        print(">>>\n", filter_expr, "\n<<<")
    return classifications


class FilterExpressionAnalyzer(JEXLAnalyzer):
    def generic_visit(self, expression):
        print("Unknown expression", expression)
        return {"unknown"}

    def visit_Literal(self, expression):
        if expression.value:
            return {"always"}
        else:
            return {"never"}

    def visit_BinaryExpression(self, expression):
        if expression.operator.symbol == "&&":
            classifications = set()
            for child in expression.children:
                classifications |= self.visit(child)
            if any(c == "never" for c in classifications):
                return {"never"}
            else:
                return set(c for c in classifications if c != "always")

        elif expression.operator.symbol == "||":
            return {"unclassified-or"}

        elif expression.operator.symbol == "==":
            if isinstance(expression.left, parser.Identifier):
                # path = id_to_context_path
                if id_to_context_path(expression.left) == ["channel"] and isinstance(
                    expression.right, parser.Literal
                ):
                    return {"channel"}
                if isinstance((expression.left, parser.Identifier)):
                    pass

            print(f"Unclassified equality: {expression}")
            return {"unclassified-equality"}

        else:
            print(f"Unknown binary expression {expression.operator.symbol}")
            return {"unknown"}

    def visit_UnaryExpression(self, expression):
        if expression.operator.symbol == "!":
            return {"unclassified-not"}

        else:
            print(f"Unknown unary expression {expression.operator.symbol}")
            return {"unknown"}

    def visit_FilterExpression(self, expression):
        if isinstance(expression.expression, parser.Literal):
            return self.visit(
                parser.Identifier(
                    value=expression.expression.value, subject=expression.subject
                )
            )
        else:
            return {"unclassified-filter-expression"}

    def visit_Identifier(self, expression):
        path = id_to_context_path(expression)
        if path[0] == "telemetry":
            return {"unclassified-telemetry"}
        else:
            return {"unclassified-context"}

    def visit_Transform(self, expression):
        method_name = f"visit_Transform_{expression.name}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(expression)
        else:
            return {f"unknown-transform-{expression.name}"}

    def visit_Transform_bucketSample(self, expression):
        if (
            all(
                isinstance(a, (parser.Identifier, parser.Literal))
                for a in expression.args
            )
            and isinstance(expression.subject, parser.ArrayLiteral)
            and all(
                isinstance(s, (parser.Identifier, parser.Literal))
                for s in expression.subject.value
            )
        ):
            return {"simple-bucket-sample"}
        else:
            return {"complex-bucket-sample"}


def id_to_context_path(identifier):
    rv = []
    pointer = identifier
    while pointer:
        rv.append(pointer.value)
        pointer = pointer.subject
    rv.reverse()
    if rv[0] in ["normandy", "env"]:
        return rv[1:]
    raise Exception(f"Unknown context root {rv[0]}")
