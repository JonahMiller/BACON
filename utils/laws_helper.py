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
        self.nonlinear = False

    def compute(self):
        if self.nonlinear:
            print("Unable to deal with non-linear relationship, returning blank")
            return None, None, None

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
            if terms == 1:
                if len(Add.make_args(den)) == 2:
                    return self.complex_linear_1_term(eqn)
            elif terms == 2 or terms == 3:
                self.eliminate_smaller_coeffs(eqn)
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
        print([coeff], self.subs_expr(expr), lin_data)
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

        # TODO: check if below is working as expected.
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

    def eliminate_smaller_coeffs(self, eqn):
        '''
        Assumes form of eta = a*nu + b + c/(nu + d)
        '''
        arg_list = Add.make_args(eqn.rhs)
        b = 0
        for arg in arg_list:
            num, den = fraction(arg)
            if len(Add.make_args(den)) == 1:
                if isinstance(arg, Number):
                    b = arg
                else:
                    a = 1
                    for ar in list(arg.args):
                        if isinstance(ar, Number):
                            a = ar
            else:
                c = num
                bot_coeff = 1
                for arg in list(den.args):
                    if isinstance(arg, Number):
                        d = arg
                    else:
                        for ar in list(arg.args):
                            if isinstance(ar, Number):
                                bot_coeff = ar

        a, b, c, d = a, b, c/bot_coeff, d/bot_coeff

        if abs(d) < 0.0001:
            d = 0
        if b == 0:
            b = 1e-15

        if abs(a/c) > 100 or abs(b/c) > 100:
            if abs(a/b) > 100:
                self.rhs = a*nu
            elif abs(b/a) > 100:
                self.rhs = b
            else:
                self.rhs = a*nu + b
        elif abs(c/a) > 100:
            if abs(c/b) < 100 and d == 0:
                self.rhs = b + c/(nu)
            elif abs(c/b) > 100:
                self.rhs = c/(nu + d)
        else:
            self.nonlinear = True
        self.compute()

    def subs_expr(self, expr):
        return expr.subs(eta, self.symbols[0]).subs(nu, self.symbols[1])
