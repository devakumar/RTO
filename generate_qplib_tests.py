import os
import numpy as np
from problem_classes.qplib import QPLIB

def get_qplib_problem(ID = '0018', path='problem_classes/qplib_data'):
	loc = path + os.sep + 'QPLIB_' + ID + '.qplib'

	print(loc)
	if os.path.exists(loc):
		problem = QPLIB(loc).qp_problem
	else :
		problem = dict()
	
	return problem

def to_string(P):
	#S = np.array2string(P, separator=',', prefix="\t\t", threshold=5000,
	#formatter={'float_kind':lambda x: "%.4f" % x})

	S = np.array2string(P, precision=4, separator=',', prefix="\t\t\t\t\t\t", max_line_width=1e10, threshold=1e5)

	return S.replace("]", "").replace("[", "")

HEADER = """#define QP_SOLVER_PRINTING
#include "gtest/gtest.h"
#include "qp_solver.hpp"
#include <Eigen/IterativeLinearSolvers>

using namespace qp_solver; 
const double inf = std::numeric_limits<double>::infinity();\n"""

def get_qp_problem(problem, QPname):
	P = to_string(problem['P'].A)
	q = to_string(problem['q'])
	A = to_string(problem['A'].A)
	l = to_string(problem['l'])
	u = to_string(problem['u'])
	n, m = problem['n'], problem['m']
	Problem = """\ntemplate <typename _Scalar=double>
class _{0} : public QP<{1}, {2}, _Scalar>
{{
public:
	_{0}()
	{{
		this->P << {3};
		this->q << {4};
		this->A << {5};
		this->l << {6};
		this->u << {7};
	}}
}};\n""".format(QPname, n, m, P, q, A, l, u)

	return Problem

def get_qp_problem_from_file(problem, QPname, writeTo = "qplib"):
	P = problem['P'].A
	q = problem['q']
	A = problem['A'].A
	l = problem['l']
	u = problem['u']
	n, m = problem['n'], problem['m']

	folder = writeTo + os.sep + QPname
	if not os.path.exists(folder):
		os.system("mkdir -p {}".format(folder))
	
	Mats = ['P', 'q', 'A', 'l', 'u']
	files = [folder + os.sep + M + ".csv" for M in Mats]
	for i, M in enumerate([P, q, A, l, u]):
		np.savetxt(files[i], M, delimiter=',', fmt='%.4f')
	Problem = """\ntemplate <typename _Scalar=double>
class _{0} : public QP<{1}, {2}, _Scalar>
{{
public:
	_{0}()
	{{
		Eigen::MatrixXd P = load_csv<Eigen::MatrixXd>("{3}");
		Eigen::MatrixXd q = load_csv<Eigen::MatrixXd>("{4}");
		Eigen::MatrixXd A = load_csv<Eigen::MatrixXd>("{5}");
		Eigen::MatrixXd l = load_csv<Eigen::MatrixXd>("{6}");
		Eigen::MatrixXd u = load_csv<Eigen::MatrixXd>("{7}");
		this->P = P;
		this->q = q;
		this->A = A;
		this->l = l;
		this->u = u;
	}}
}};\n""".format(QPname, n, m, files[0], files[1], files[2], files[3], files[4])

	return Problem

def test_QP(QPname):
	TEST = """\nusing {0} = _{0}<double>;

TEST(QPProblemSets, testQP{0}default) {{
	{0} qp;
	QPSolver<{0}> prob;

	prob.settings().max_iter = 4000;
	prob.settings().verbose = true;
	prob.settings().alpha = 1.6;
	prob.settings().adaptive_rho = false;
	prob.settings().check_termination = 25;

	prob.setup(qp);
	prob.solve(qp);
	Eigen::VectorXd sol = prob.primal_solution();

	EXPECT_LT(prob.iter, prob.settings().max_iter);
	EXPECT_EQ(prob.info().status, SOLVED);
	// check feasibility (with some epsilon margin)
	Eigen::VectorXd lower = qp.A*sol - qp.l;
	Eigen::VectorXd upper = qp.A*sol - qp.u;
	EXPECT_GE(lower.minCoeff(), -1e-3);
	EXPECT_LE(upper.maxCoeff(), 1e-3);
}}

TEST(QPProblemSets, testQP{0}adaptive) {{
	{0} qp;
	QPSolver<{0}> prob;

	prob.settings().max_iter = 4000;
	prob.settings().verbose = true;
	prob.settings().alpha = 1.6;
	prob.settings().adaptive_rho = true;
	prob.settings().adaptive_rho_interval = 25;
	prob.settings().check_termination = 25;

	prob.setup(qp);
	prob.solve(qp);
	Eigen::VectorXd sol = prob.primal_solution();

	EXPECT_LT(prob.iter, prob.settings().max_iter);
	EXPECT_EQ(prob.info().status, SOLVED);
	// check feasibility (with some epsilon margin)
	Eigen::VectorXd lower = qp.A*sol - qp.l;
	Eigen::VectorXd upper = qp.A*sol - qp.u;
	EXPECT_GE(lower.minCoeff(), -1e-3);
	EXPECT_LE(upper.maxCoeff(), 1e-3);
}} \n""".format(QPname)

	return TEST

if __name__ == "__main__":
	#Load problem from QPLIB
	cases = ["0018"]#, "0343"]#, "2712"]

	test_cases = []

	for ID in cases:
		problem = get_qplib_problem(ID)

		QPname = "QP" + ID
		# Convert problem to polympc format
		qp_problem = get_qp_problem_from_file(problem, QPname)
		#qp_problem = get_qp_problem(problem, QPname)

		test = test_QP(QPname)

		test_cases.extend([qp_problem, test])

	with open("qplib_problem_sets.cpp", 'w') as f:
		lines = [HEADER]
		lines.extend(test_cases)
		f.writelines(lines)
