package simplex

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestSimplex(t *testing.T) {
	// create new example matrix
	maximize := mat.NewVecDense(3, []float64{5, 4, 3})

	constraints := mat.NewDense(3, 4, []float64{
		2, 3, 1, 5,
		4, 1, 2, 11,
		3, 4, 2, 8})

	actualResult := Solve(maximize, constraints)
	var expectedResult float64 = 13

	if actualResult != expectedResult {
		t.Fatalf("Expected %f but got %f", expectedResult, actualResult)
	}
}
