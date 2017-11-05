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
	initialVars := cols - 1
	constraints := rows - 1

	fmt.Printf("We have %d initial variables.\n", initialVars)
	fmt.Printf("We have %d constraints.\n", constraints)
	result := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows-1; i++ {
		row := problem.RowView(i)
		// set the last element as the first
		result.Set(i, 0, row.At(row.Len()-1, 0))

		for j := 0; j < row.Len()-1; j++ {
			value := row.At(j, 0)
			if i > 0 {
				value *= -1
			}
			result.Set(i, j+1, value)
		}
	}

	// display result
	resultRows, _ := result.Dims()
	for i := 0; i < resultRows; i++ {
		fmt.Println(result.RawRowView(i))
	}
}
