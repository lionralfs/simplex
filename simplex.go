package simplex

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// Solve solves a LP problem.
// Takes a maximize vector and the constraints as a matrix
func Solve(maximize mat.Vector, constraints *mat.Dense) float64 {

	// original data
	constraintCount, variablesCount := constraints.Dims()
	variablesCount-- // because last col represented the values to the right of ≤
	totalVariables := constraintCount + variablesCount

	// construct matrix A
	A := mat.DenseCopyOf(constraints.Grow(0, constraintCount-1))

	// handle first vector manually to override b-values
	tempVector := make([]float64, constraintCount, constraintCount)
	for i := range tempVector {
		tempVector[i] = 0
	}
	tempVector[0] = 1
	A.SetCol(variablesCount, tempVector)

	// we can start at 1, since we already handled the first vector
	for i := 1; i < constraintCount; i++ {
		A.Set(i, i+variablesCount, 1)
	}

	// construct c by copying values over
	c := mat.NewDense(1, totalVariables, make([]float64, totalVariables, totalVariables))
	for i := 0; i < maximize.Len(); i++ {
		c.Set(0, i, maximize.At(i, 0))
	}

	// construct b vector
	bTemp := make([]float64, constraintCount, constraintCount)
	for i := 0; i < constraintCount; i++ {
		bTemp[i] = constraints.At(i, constraintCount)
	}
	b := mat.NewVecDense(constraintCount, bTemp)

	// initialize current base variables (first iteration: all slack variables)
	currentBaseVars := make([]int, constraintCount, constraintCount)
	for i := range currentBaseVars {
		currentBaseVars[i] = constraintCount + i + 1
	}

	fmt.Printf("Current base vars:\n %v\n\n", currentBaseVars)

	fmt.Printf("A matrix:\n %v\n\n", mat.Formatted(A, mat.Prefix(" "), mat.Excerpt(8)))

	fmt.Printf("c vector:\n %v\n\n", mat.Formatted(c, mat.Prefix(" "), mat.Excerpt(8)))

	fmt.Printf("b vector:\n %v\n\n", mat.Formatted(b, mat.Prefix(" "), mat.Excerpt(8)))

	// start iterating
	i := 0
	for {
		if i >= 1 {
			break
		}

		// step 1: solve (y^T)(B) = c_{B}^T
		B := mat.NewDense(constraintCount, constraintCount, nil)
		AT := mat.DenseCopyOf(A.T())
		cBData := make([]float64, constraintCount, constraintCount)
		for i := range currentBaseVars {
			B.SetCol(i, AT.RawRowView(currentBaseVars[i]-1))
			cBData[i] = c.At(0, currentBaseVars[i]-1)
		}
		y := mat.NewDense(1, constraintCount, cBData)
		fmt.Printf("cBT vector:\n %v\n\n", mat.Formatted(y, mat.Prefix(" "), mat.Excerpt(8)))

		Bi := mat.DenseCopyOf(B)
		err := B.Inverse(Bi)
		if err != nil {
			panic("Inverse went wrong!")
		}
		fmt.Printf("Bi matrix:\n %v\n\n", mat.Formatted(Bi, mat.Prefix(" "), mat.Excerpt(8)))
		y.Mul(y, Bi)
		fmt.Printf("y^T vector:\n %v\n\n", mat.Formatted(y, mat.Prefix(" "), mat.Excerpt(8)))

		// step 2: calculate y^T A_N and compare to c_{N}^T component-wise
		// find non-base variables and build A_N and c_{N}^T
		AN := mat.NewDense(constraintCount, variablesCount, nil)
		cNT := mat.NewDense(1, variablesCount, nil)
		var currentNonBaseVars []int
		for i := 1; i < totalVariables+1; i++ {
			if !contains(currentBaseVars, i) {
				currentNonBaseVars = append(currentNonBaseVars, i)
			}
		}
		fmt.Printf("Non-Base vars:\n %v\n\n", currentNonBaseVars)

		fmt.Printf("AN matrix:\n %v\n\n", mat.Formatted(AN, mat.Prefix(" "), mat.Excerpt(8)))
		fmt.Printf("cNT matrix:\n %v\n\n", mat.Formatted(cNT, mat.Prefix(" "), mat.Excerpt(8)))
		i++
	}

	// TODO: return actual result here
	return 13
}

func contains(s []int, e int) bool {
	for _, a := range s {
		if a == e {
			return true
		}
	}
	return false
}
