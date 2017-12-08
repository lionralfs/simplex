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
	totalVariables := constraintCount + variablesCount - 1

	// construct matrix A
	A := mat.DenseCopyOf(constraints.Grow(0, constraintCount-1))

	// handle first vector manually to override b-values
	tempVector := make([]float64, constraintCount, constraintCount)
	for i := range tempVector {
		tempVector[i] = 0
	}
	tempVector[0] = 1
	A.SetCol(variablesCount-1, tempVector)

	// we can start at 1, since we already handled the first vector
	for i := 1; i < constraintCount; i++ {
		A.Set(i, i+variablesCount-1, 1)
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

	fmt.Printf("A matrix:\n %v\n\n", mat.Formatted(A, mat.Prefix(" "), mat.Excerpt(8)))

	fmt.Printf("c vector:\n %v\n\n", mat.Formatted(c, mat.Prefix(" "), mat.Excerpt(8)))

	fmt.Printf("b vector:\n %v\n\n", mat.Formatted(b, mat.Prefix(" "), mat.Excerpt(8)))

	// TODO: return actual result here
	return 0
}
