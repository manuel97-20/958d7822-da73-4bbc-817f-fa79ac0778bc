import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Cashflow(object):
   def __init__(self, amount, t):
        self.amount = amount
        self.t = t

   def value_at(self, t, interest_rate):
        delta_t = t - self.t
        return self.amount * (1 + interest_rate) ** delta_t

    def present_value(self,interest_rate):
        return self.value_at(t=0,interest_rate=interest_rate)


class InvestmentProject(object):
    RISK_FREE_RATE = 0.08

    def __init__(self, cashflows, hurdle_rate=RISK_FREE_RATE):
        cashflows_positions = {str(flow.t): flow for flow in cashflows}
        self.cashflow_max_position = max((flow.t for flow in cashflows))
        self.cashflows = []
        for t in range(self.cashflow_max_position + 1):
            self.cashflows.append(cashflows_positions.get(str(t), Cashflow(t=t, amount=0)))
        self.hurdle_rate = hurdle_rate if hurdle_rate else InvestmentProject.RISK_FREE_RATE

    @staticmethod
    def from_csv(filepath, hurdle_rate=RISK_FREE_RATE):
        cashflows = [Cashflow(**row) for row in pd.read_csv(filepath).T.to_dict().values()]
        return InvestmentProject(cashflows=cashflows, hurdle_rate=hurdle_rate)

    @property
    def internal_return_rate(self):
        return np.irr([flow.amount for flow in self.cashflows])


    def plot(self, show=False):
        if show:
            df = pd.DataFrame(self.cashflows)
            plot = df.plot.bar(x="t", y="amount", stacked=True)
            fig = plot.get_figure()
        else:
            df = pd.DataFrame(self.cashflows)
            plot = df.plot.bar(x="t", y="amount", stacked=True)
            fig = plot.get_figure()
            plt.show()
            return fig

    def net_present_value(self, interest_rate=None):
        npv =  np.npv(interest_rate,[self.cashflows])
        return npv

    def equivalent_annuity(self, interest_rate=None):
        if interest_rate is None:
            interest_rate = self.hurdle_rate

        c = (interest_rate*self.net_present_value())/(1-(1+interest_rate)**(-self.cashflow_max_position))
        return c

    def describe(self):
        return {
            "irr": self.internal_return_rate,
            "hurdle-rate": self.hurdle_rate,
            "net-present-value": self.net_present_value(interest_rate=None),
            "equivalent-annuity": self.equivalent_annuity(interest_rate=None)
        }
