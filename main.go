package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func main() {
	// create new example matrix
	a := mat.NewDense(4, 4, []float64{5, 4, 3, 0, 2, 3, 1, 5, 4, 1, 2, 11, 3, 4, 2, 8})

	solve(a)
}

func solve(problem *mat.Dense) {
	rows, cols := problem.Dims()
	result := mat.NewDense(rows, cols, nil)

	// add slack variables
	for i := 1; i < rows; i++ {
		result.Set(i, cols-1, problem.At(i, cols-1))
		for j := 0; j < cols-1; j++ {
			fmt.Println(problem.At(i, j))
		}
	}

	fmt.Println(*result)
}
