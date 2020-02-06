from pyjexl.analysis import JEXLAnalyzer
from pyjexl.jexl import JEXL
from pyjexl import parser

from .log import log


def classify_filter(filter_expr):
    jexl = JEXL()
    classifications = jexl.analyze(filter_expr, FilterExpressionAnalyzer)
    if "unknown" in classifications:
        log.info(">>>\n", filter_expr, "\n<<<")
    return classifications


class FilterExpressionAnalyzer(JEXLAnalyzer):
    def generic_visit(self, expression):
        log.info("Unknown expression", expression)
        return {"unknown"}

    def visit_Literal(self, expression):
        if expression.value:
            return {"always"}
        else:
            return {"never"}

    def visit_BinaryExpression(self, expression):
        simple_filters = ["locale", "channel", "country", "distribution", "version"]

        if expression.operator.symbol == "&&":
            classifications = set()
            for child in expression.children:
                classifications |= self.visit(child)
            if any(c == "never" for c in classifications):
                return {"never"}
            else:
                return set(c for c in classifications if c != "always")

        elif expression.operator.symbol == "||":
            return {"unclassified", "unclassified-or"}

        elif expression.operator.symbol == "==":
            if isinstance(expression.left, parser.Identifier) and isinstance(
                expression.right, parser.Literal
            ):
                path, classifications = id_to_context_path(expression.left)
                for key in simple_filters:
                    if path == [key] and isinstance(expression.right, parser.Literal):
                        return classifications | {key}
                if path[0] == "telemetry":
                    return classifications | {"unclassified", "unclassified-telemetry"}
            elif isinstance(expression.left, parser.Transform) and isinstance(
                expression.right, parser.Literal
            ):
                if expression.left.name == "preferenceValue":
                    return {"preference-value-eq"}
                elif expression.left.name == "preferenceExists":
                    return {"preference-exists"}

            # log.info(f"Unclassified equality: {expression}")
            return {"unclassified", "unclassified-equality"}

        elif expression.operator.symbol == "!=":
            return {"unclassified", "unclassified-ne"}

        elif expression.operator.symbol in [">=", ">", "<", "<="]:
            if isinstance(expression.left, parser.Identifier) and isinstance(
                expression.right, parser.Literal
            ):
                path, classifications = id_to_context_path(expression.left)
                if path == "version":
                    return classifications | {"version"}
                return classifications | {
                    "unclassified",
                    "unclassified-simple-comparison",
                }
            return {"unclassified", "unclassified-cmp"}

        elif expression.operator.symbol == "in":
            if isinstance(expression.left, parser.Identifier):
                path, classifications = id_to_context_path(expression.left)
                for key in simple_filters:
                    if path == [key] and isinstance(
                        expression.right, parser.ArrayLiteral
                    ):
                        return classifications | {f"{key}-in-list"}

                if path == ["searchEngine"]:
                    return classifications | {"search-engine-in-list"}

                if path[0] == "telemetry":
                    return classifications | {"telemetry-value-in-list"}

            elif isinstance(expression.right, parser.Identifier):
                path, classifications = id_to_context_path(expression.right)
                if path[0] == "telemetry":
                    return classifications | {"unclassified", "unclassified-telemetry"}
                if path[0] == "experiments":
                    return classifications | {"experiments"}
                if path == ["searchEngine"]:
                    return classifications | {"search-engine"}
                if path == ["addons"]:
                    return classifications | {"bug-in-with-addons"}

            if (
                isinstance(expression.left, parser.Transform)
                and expression.left.name == "preferenceValue"
                and isinstance(expression.right, parser.ArrayLiteral)
                and all(isinstance(el, parser.Literal) for el in expression.right.value)
            ):
                return {"pref-value-in-list"}

            log.info(f"Unclassified in: {expression}")
            return {"unclassified", "unclassified-in"}

        else:
            log.info(f"Unknown binary expression {expression.operator.symbol}")
            return {"unknown"}

    def visit_UnaryExpression(self, expression):
        if expression.operator.symbol == "!":
            inner = self.visit(expression.right)
            return {"not"} | {f"not-{c}" for c in inner}

        else:
            log.info(f"Unknown unary expression {expression.operator.symbol}")
            return {"unknown"}

    def visit_FilterExpression(self, expression):
        if isinstance(expression.expression, parser.Literal):
            return self.visit(
                parser.Identifier(
                    value=expression.expression.value, subject=expression.subject
                )
            )
        else:
            return {"unclassified", "unclassified-filter-expression"}

    def visit_Identifier(self, expression):
        path, classifications = id_to_context_path(expression)
        if path[0] == "telemetry":
            return classifications | {"unclassified", "unclassified-telemetry"}
        elif path == ["isFirstRun"]:
            return classifications | {"first-run"}
        else:
            return classifications | {"unclassified", "unclassified-context"}

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

    def visit_Transform_stableSample(self, expression):
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
            return {"simple-stable-sample"}
        else:
            return {"complex-stable-sample"}

    def visit_Transform_preferenceValue(self, expression):
        return {"preference-value"}


def id_to_context_path(identifier):
    path = []
    classifications = set()

    pointer = identifier
    while pointer:
        if isinstance(pointer, parser.FilterExpression):
            if isinstance(pointer.expression, parser.Literal):
                path.append(pointer.expression.value)
            else:
                path.append("SOMETHING WEIRD")
        elif isinstance(pointer, parser.Identifier):
            path.append(pointer.value)
        pointer = pointer.subject

    path.reverse()
    if path[0] in ["normandy", "env"]:
        path = path[1:]
    else:
        classifications.add("bug-invalid-context-root")

    return path, classifications
