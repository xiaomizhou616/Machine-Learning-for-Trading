"""MC1-P1: Analyze a portfolio - grading script.

Usage:
- Switch to a student feedback directory first (will write "points.txt" and "comments.txt" in pwd).
- Run this script with both ml4t/ and student solution in PYTHONPATH, e.g.:
    PYTHONPATH=ml4t:MC1-P1/jdoe7 python ml4t/mc1_p1_grading/grade_analysis.py

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import pytest
from grading.grading import grader, GradeResult, run_with_timeout, IncorrectOutput

import os
import sys
import traceback as tb

import pandas as pd
from collections import namedtuple, OrderedDict

from util import get_data

import datetime

# Student code
# Spring '16 renamed package to just "analysis" (BPH)
main_code = "analysis"  # module name to import

# Test cases
# Spring '16 test cases only check sharp ratio, avg daily ret, and cum_ret (BPH)
PortfolioTestCase = namedtuple('PortfolioTestCase', ['inputs', 'outputs', 'description'])
portfolio_test_cases = [
    PortfolioTestCase(
        inputs=dict(
            start_date='2010-01-01',
            end_date='2010-12-31',
            symbol_allocs=OrderedDict([('GOOG', 0.2), ('AAPL', 0.3), ('GLD', 0.4), ('XOM', 0.1)]),
            start_val=1000000),
        outputs=dict(
            cum_ret=0.255646784534,
            avg_daily_ret=0.000957366234238,
            sharpe_ratio=1.51819243641),
        description="Wiki example 1"
    ),
    PortfolioTestCase(
        inputs=dict(
            start_date='2010-01-01',
            end_date='2010-12-31',
            symbol_allocs=OrderedDict([('AXP', 0.0), ('HPQ', 0.0), ('IBM', 0.0), ('HNZ', 1.0)]),
            start_val=1000000),
        outputs=dict(
            cum_ret=0.198105963655,
            avg_daily_ret=0.000763106152672,
            sharpe_ratio=1.30798398744),
        description="Wiki example 2"
    ),
    PortfolioTestCase(
        inputs=dict(
            start_date='2010-06-01',
            end_date='2010-12-31',
            symbol_allocs=OrderedDict([('GOOG', 0.2), ('AAPL', 0.3), ('GLD', 0.4), ('XOM', 0.1)]),
            start_val=1000000),
        outputs=dict(
            cum_ret=0.205113938792,
            avg_daily_ret=0.00129586924366,
            sharpe_ratio=2.21259766672),
        description="Wiki example 3: Six month range"
    ),
    PortfolioTestCase(
        inputs=dict(
            start_date='2010-01-01',
            end_date='2013-05-31',
            symbol_allocs=OrderedDict([('AXP', 0.3), ('HPQ', 0.5), ('IBM', 0.1), ('GOOG', 0.1)]),
            start_val=1000000),
        outputs=dict(
            cum_ret=-0.110888530433,
            avg_daily_ret=-6.50814806831e-05,
            sharpe_ratio=-0.0704694718385),
        description="Normalization check"
    ),
    PortfolioTestCase(
        inputs=dict(
            start_date='2010-01-01',
            end_date='2010-01-31',
            symbol_allocs=OrderedDict([('AXP', 0.9), ('HPQ', 0.0), ('IBM', 0.1), ('GOOG', 0.0)]),
            start_val=1000000),
        outputs=dict(
            cum_ret=-0.0758725033871,
            avg_daily_ret=-0.00411578300489,
            sharpe_ratio=-2.84503813366),
        description="One month range"
    ),
    PortfolioTestCase(
        inputs=dict(
            start_date='2011-01-01',
            end_date='2011-12-31',
            symbol_allocs=OrderedDict([('WFR', 0.25), ('ANR', 0.25), ('MWW', 0.25), ('FSLR', 0.25)]),
            start_val=1000000),
        outputs=dict(
            cum_ret=-0.686004563165,
            avg_daily_ret=-0.00405018240566,
            sharpe_ratio=-1.93664660013),
        description="Low Sharpe ratio"
    ),
    PortfolioTestCase(
        inputs=dict(
            start_date='2010-01-01',
            end_date='2010-12-31',
            symbol_allocs=OrderedDict([('AXP', 0.0), ('HPQ', 1.0), ('IBM', 0.0), ('HNZ', 0.0)]),
            start_val=1000000),
        outputs=dict(
            cum_ret=-0.191620333598,
            avg_daily_ret=-0.000718040989619,
            sharpe_ratio=-0.71237182415),
        description="All your eggs in one basket"
    ),
    PortfolioTestCase(
        inputs=dict(
            start_date='2006-01-03',
            end_date='2008-01-02',
            symbol_allocs=OrderedDict([('MMM', 0.0), ('MO', 0.9), ('MSFT', 0.1), ('INTC', 0.0)]),
            start_val=1000000),
        outputs=dict(
            cum_ret=0.43732715979,
            avg_daily_ret=0.00076948918955,
            sharpe_ratio=1.26449481371),
        description="Two year range"
    ),
    PortfolioTestCase(
        inputs=dict(
            start_date='2010-03-01',
            end_date='2011-03-01',
            symbol_allocs=OrderedDict([('ABC', 0.05), ('AMAT', 0.05), ('BHI', 0.05), ('FAST', 0.85)]),
            start_val=1000000,
            risk_free_rate=0.0),
        outputs=dict(
            cum_ret=0.371025877148,
            avg_daily_ret=0.00138367240405,
            sharpe_ratio=1.33178048758),
        description="bpotocki3 - Test 1 - Different stocks"
    ),
    PortfolioTestCase(
        inputs=dict(
            start_date='2010-01-01',
            end_date='2010-12-31',
            symbol_allocs=OrderedDict([('AXP', 0.0), ('HPQ', 1.0), ('IBM', 0.0), ('HNZ', 0.0)]),
            start_val=1000000,
            risk_free_rate = 0.01),
        outputs=dict(
            cum_ret=-0.191620333598,
            avg_daily_ret=-0.000718040989619,
            sharpe_ratio=-10.6334185951),
        description="bpotocki3 - Test 2 - All your eggs in one basket (but different rfr)"
    ),
    PortfolioTestCase(
        inputs=dict(
            start_date='2009-01-01',
            end_date='2009-12-31',
            symbol_allocs=OrderedDict([('HRS', 0.4), ('HOG', 0.6)]),
            start_val=1000000,
            risk_free_rate = 0.0),
        outputs=dict(
            cum_ret=0.325388695131,
            avg_daily_ret=0.00165153516639,
            sharpe_ratio=0.803568833348),
        description="bpotocki3 - Test 3 - Two stocks"
    ),
    PortfolioTestCase(
        inputs=dict(
            start_date='2009-01-01',
            end_date='2009-12-31',
            symbol_allocs=OrderedDict([('YUM', 0.4), ('T', 0.2), ('PEP', 0.16), ('NKE', 0.1), ('JDSU', 0.14)]),
            start_val=1000000,
            risk_free_rate = 0.0),
        outputs=dict(
            cum_ret=0.23667066654,
            avg_daily_ret=0.00100977667535,
            sharpe_ratio=0.885771354148),
        description="bpotocki3 - Test 4 - Five stocks"
    ),
    PortfolioTestCase(
        inputs=dict(
            start_date='2010-01-01',
            end_date='2010-12-31',
            symbol_allocs=OrderedDict([('IBM', 0.9), ('AAPL', 0.025), ('GOOG', 0.025), ('AMZN', 0.025), ('MSFT', 0.025)]),
            start_val=100000,
            risk_free_rate = 0.0),
        outputs=dict(
            cum_ret=0.134272143707,
            avg_daily_ret=0.000563274404767,
            sharpe_ratio=0.806551033542),
        description="Test - different allocations"
    ),
    PortfolioTestCase(
        inputs=dict(
            start_date='2010-01-01',
            end_date='2010-12-31',
            symbol_allocs=OrderedDict([('IBM', 0.9), ('AAPL', 0.025), ('GOOG', 0.025), ('AMZN', 0.025), ('MSFT', 0.025)]),
            start_val=100000,
            risk_free_rate = 0.08),
        outputs=dict(
            cum_ret=0.134272143707,
            avg_daily_ret=0.000563274404767,
            sharpe_ratio=-113.745223621),
        description="Test - high rfr"
    ),
    PortfolioTestCase(
        inputs=dict(
            start_date='2011-05-14',
            end_date='2011-11-12',
            symbol_allocs=OrderedDict(
                [('ABI', 0.086431226765799257), ('BDK', 0.088289962825278803), ('SCHW', 0.011152416356877323),
                 ('SYMC', 0.0092936802973977699), ('MTW', 0.077137546468401486), ('AXP', 0.031598513011152421),
                 ('SLE', 0.01579925650557621), ('CTSH', 0.079925650557620811), ('FE', 0.010223048327137546),
                 ('ACS', 0.038104089219330853), ('SWN', 0.024163568773234202), ('ATI', 0.065985130111524168),
                 ('DGX', 0.045539033457249072), ('MSI', 0.0037174721189591081), ('MRK', 0.085501858736059491),
                 ('NOV', 0.067843866171003714), ('TROW', 0.061338289962825282), ('HON', 0.070631970260223054),
                 ('CLF', 0.067843866171003714), ('TAP', 0.059479553903345729)]),
            start_val=473210,
            risk_free_rate=0.5),
        outputs=dict(
            cum_ret=-0.0998946557467,
            avg_daily_ret=-0.000518843039236,
            sharpe_ratio=-316.076050529),
        description='rhunter: testcase 1'
    ),
    PortfolioTestCase(
        inputs=dict(
            start_date='2000-02-01',
            end_date='2012-09-12',
            symbol_allocs=OrderedDict(
                [(u'ABI', 0.049792531120331947), (u'RRD', 0.029045643153526972), (u'KLAC', 0.17634854771784231),
                (u'PCL', 0.070539419087136929), (u'JNY', 0.045643153526970952), (u'JWN', 0.15767634854771784),
                (u'PCP', 0.13900414937759337), (u'CVH', 0.17634854771784231), (u'JAVA', 0.14937759336099585),
                (u'PTV', 0.0062240663900414933)]),
            start_val=1520812,
            risk_free_rate=0.44),
        outputs=dict(
            cum_ret=6.67456115008,
            avg_daily_ret=0.000815479791537,
            sharpe_ratio=-376.886136516),
        description='rhunter: testcase 4'
    ),
    PortfolioTestCase(
        inputs=dict(
            start_date='2000-02-01',
            end_date='2012-09-12',
            symbol_allocs=OrderedDict(
                [(u'MDT', 0.06699346405228758), (u'MWW', 0.12254901960784316), (u'CHK', 0.11928104575163399),
                 (u'FDX', 0.10130718954248367), (u'VAR', 0.09477124183006537), (u'V', 0.16339869281045755),
                (u'ESRX', 0.13398692810457516), (u'ATI', 0.017973856209150329), (u'BDK', 0.047385620915032685),
                (u'WB', 0.13235294117647062)]),
            start_val=1677213,
            risk_free_rate=0.06),
        outputs=dict(
            cum_ret=4.04546807449,
            avg_daily_ret=0.000652321143352,
            sharpe_ratio=-55.9887845706),
        description='rhunter: testcase 5'
    ),
    PortfolioTestCase(
        inputs=dict(
            start_date='2000-02-01',
            end_date='2012-09-12',
            symbol_allocs=OrderedDict(
                [(u'DLTR', 0.021594684385382059), (u'NBR', 0.13122923588039867), (u'UNP', 0.1395348837209302),
                (u'AIG', 0.10631229235880398), (u'M', 0.078073089700996662), (u'FLIR', 0.083056478405315604),
                (u'X', 0.1395348837209302), (u'SPY', 0.088039867109634545), (u'PTV', 0.096345514950166106),
                (u'BDK', 0.11627906976744184)]),
            start_val=1910576,
            risk_free_rate=0.26),
        outputs=dict(
            cum_ret=1.77409813034,
            avg_daily_ret=0.000460208201883,
            sharpe_ratio=-247.490958542),
        description='rhunter: testcase 6'
    ),
    PortfolioTestCase(
        inputs=dict(
            start_date='2009-01-15',
            end_date='2009-02-15',
            symbol_allocs=OrderedDict(
                [(u'FAKE1', 0.5), (u'FAKE2', 0.5)]),
            start_val=1000,
            risk_free_rate=0.03),
        outputs=dict(
            cum_ret=0.0630331753555,
            avg_daily_ret=0.00389674472,
            sharpe_ratio=-9.90594355182),
        description='Test - fake data'
    )
]
abs_margins = dict(cum_ret=0.001, avg_daily_ret=0.00001, sharpe_ratio=0.001)  # absolute margin of error for each output
points_per_output = dict(cum_ret=2.5, avg_daily_ret=2.5, sharpe_ratio=5.0)  # points for each output, for partial credit
points_per_test_case = sum(points_per_output.values())
max_seconds_per_call = 5

# Grading parameters (picked up by module-level grading fixtures)
max_points = float(len(portfolio_test_cases) * points_per_test_case)
html_pre_block = True  # surround comments with HTML <pre> tag (for T-Square comments field)

# Test functon(s)
@pytest.mark.parametrize("inputs,outputs,description", portfolio_test_cases)
def test_analysis(inputs, outputs, description, grader):
    """Test get_portfolio_value() and get_portfolio_stats() return correct values.

    Requires test inputs, expected outputs, description, and a grader fixture.
    """

    points_earned = 0.0  # initialize points for this test case
    try:
        # Try to import student code (only once)
        if not main_code in globals():
            import importlib
            # * Import module
            mod = importlib.import_module(main_code)
            globals()[main_code] = mod

        # Unpack test case
        start_date_str = inputs['start_date'].split('-')
        start_date = datetime.datetime(int(start_date_str[0]),int(start_date_str[1]),int(start_date_str[2]))
        end_date_str = inputs['end_date'].split('-')
        end_date = datetime.datetime(int(end_date_str[0]),int(end_date_str[1]),int(end_date_str[2]))
        symbols = inputs['symbol_allocs'].keys()  # e.g.: ['GOOG', 'AAPL', 'GLD', 'XOM']
        allocs = inputs['symbol_allocs'].values()  # e.g.: [0.2, 0.3, 0.4, 0.1]
        start_val = inputs['start_val']
        risk_free_rate = inputs.get('risk_free_rate',0.0)

        # the wonky unpacking here is so that we only pull out the values we say we'll test.
        def timeoutwrapper_analysis():
            student_rv = analysis.assess_portfolio(\
                    sd=start_date, ed=end_date,\
                    syms=symbols,\
                    allocs=allocs,\
                    sv=start_val, rfr=risk_free_rate, sf=252.0, \
                    gen_plot=False)
            return student_rv
        result = run_with_timeout(timeoutwrapper_analysis,max_seconds_per_call,(),{})
        student_cr = result[0]
        student_adr = result[1]
        student_sr = result[3]
        port_stats = OrderedDict([('cum_ret',student_cr), ('avg_daily_ret',student_adr), ('sharpe_ratio',student_sr)])
        # Verify against expected outputs and assign points
        incorrect = False
        msgs = []
        for key, value in port_stats.iteritems():
            if abs(value - outputs[key]) > abs_margins[key]:
                incorrect = True
                msgs.append("    {}: {} (expected: {})".format(key, value, outputs[key]))
            else:
                points_earned += points_per_output[key]  # partial credit

        if incorrect:
            inputs_str = "    start_date: {}\n" \
                         "    end_date: {}\n" \
                         "    symbols: {}\n" \
                         "    allocs: {}\n" \
                         "    start_val: {}".format(start_date, end_date, symbols, allocs, start_val)
            raise IncorrectOutput, "One or more stats were incorrect.\n  Inputs:\n{}\n  Wrong values:\n{}".format(inputs_str, "\n".join(msgs))
    except Exception as e:
        # Test result: failed
        msg = "Test case description: {}\n".format(description)
        
        # Generate a filtered stacktrace, only showing erroneous lines in student file(s)
        tb_list = tb.extract_tb(sys.exc_info()[2])
        for i in xrange(len(tb_list)):
            row = tb_list[i]
            tb_list[i] = (os.path.basename(row[0]), row[1], row[2], row[3])  # show only filename instead of long absolute path
        tb_list = [row for row in tb_list if row[0] == 'analysis.py']
        if tb_list:
            msg += "Traceback:\n"
            msg += ''.join(tb.format_list(tb_list))  # contains newlines
        msg += "{}: {}".format(e.__class__.__name__, e.message)

        # Report failure result to grader, with stacktrace
        grader.add_result(GradeResult(outcome='failed', points=points_earned, msg=msg))
        raise
    else:
        # Test result: passed (no exceptions)
        grader.add_result(GradeResult(outcome='passed', points=points_earned, msg=None))


if __name__ == "__main__":
    pytest.main(["-s", __file__])
