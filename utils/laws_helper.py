from sympy import Eq, Add, Symbol, Number, nsimplify, solve, fraction, expand


eta = Symbol("eta")
nu = Symbol("nu")
psi = Symbol("psi")


def new_symbol(all_found_symbols):
    '''
    Draws new variables to use in the case of linear relationships. Starts with
    "a" and draws onwards in the alphabet.
    '''
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    for sym in all_found_symbols:
        try:
            idx = letters.index(str(sym))
            letters = letters[idx + 1:]
        except ValueError:
            continue
    letter = Symbol(letters[0])
    return letter


class return_equation:
    def __init__(self, eqn_rhs, symbols, all_found_symbols):
        self.rhs = eqn_rhs
        self.symbols = symbols
        self.all_found_symbols = all_found_symbols

        print(self.rhs)

    def compute(self):
        eqn = Eq(eta, self.rhs)

        rhs_symb, terms = self.rhs.count(nu), len(Add.make_args(self.rhs))

        if rhs_symb == 0:
            return [float(self.rhs.subs(nu, self.symbols[1]))], self.symbols[0], ""

        if rhs_symb == 1:
            if terms == 1:
                num, den = fraction(eqn.rhs)
                if len(Add.make_args(den)) == 1:
                    return self.single_product_division(eqn)
                else:
                    return self.complex_linear_1_term(eqn)
            elif terms == 2:
                return self.simple_linear(eqn)
            else:
                print("Unable to deal with non-linear relationship, returning blank")
                return None, None, None

        elif rhs_symb == 2:
            num, den = fraction(eqn.rhs)
            if len(Add.make_args(den)) == 2:
                if terms == 1:
                    return self.complex_linear_1_term(eqn)
            else:
                print("Unable to deal with non-linear relationship, returning blank")
                return None, None, None

        else:
            print("Unable to deal with non-linear relationship, returning blank")
            return None, None, None

    def single_product_division(self, eqn):
        arg_list = list(eqn.rhs.args)
        const = 1
        for arg in arg_list:
            if isinstance(arg, Number):
                const = arg
        expr = solve(eqn, const)[0]
        expr = self.subs_expr(expr)
        return [const], expr, None

    def simple_linear(self, eqn):
        const = 1
        for arg in list(Add.make_args(eqn.rhs)):
            if isinstance(arg, Number):
                const = arg
        expr = solve(eqn, const)[0]
        coeff, var1, var2 = self.linear_term_coeff(expr)
        k = new_symbol(self.all_found_symbols)
        lin_data = ["linear", k, var1 - nsimplify(k*var2), -coeff, var1, var2]
        return [coeff], self.subs_expr(expr), lin_data

    def complex_linear_1_term(self, eqn):
        num, den = fraction(eqn.rhs)

        if len(Add.make_args(num)) != 1:
            print("Unable to deal with non-linear relationship, returning blank")
            return None, None, None

        bot_coeff = 1
        for arg in list(den.args):
            if isinstance(arg, Number):
                const = arg
            else:
                var = arg
                for ar in list(var.args):
                    if isinstance(ar, Number):
                        bot_coeff = ar

        if abs(const/bot_coeff) < 0.0001:
            new_rhs = num/(bot_coeff*var)
            new_eqn = Eq(eqn.lhs, new_rhs)
            return self.single_product_division(new_eqn)

        expr = solve(Eq(eqn.lhs, (num/bot_coeff)/(var/bot_coeff + psi)), psi)[0]
        try:
            coeff, var1, var2 = self.linear_term_coeff(expr)
        except UnboundLocalError:
            coeff, var1, var2 = self.linear_term_coeff(-expr)

        k = new_symbol(self.all_found_symbols)
        lin_data = ["linear", k, var1 - nsimplify(k*var2), -coeff, var1, var2]
        return [const/bot_coeff], self.subs_expr(expr), lin_data

    def linear_term_coeff(self, expr):
        expr = expand(expr)
        arg_list = Add.make_args(expr)
        assert len(arg_list) == 2, f"arg list must has length 2 but instead is {arg_list}"
        var2 = arg_list[1]
        for arg in arg_list:
            if any(isinstance(term, Number) for term in list(arg.args)):
                for term in arg.args:
                    if isinstance(term, Number):
                        coeff = term
                    else:
                        var2 = term
            else:
                var1 = arg
                coeff = 1
        return coeff, self.subs_expr(var1), self.subs_expr(var2)

    def subs_expr(self, expr):
        return expr.subs(eta, self.symbols[0]).subs(nu, self.symbols[1])
