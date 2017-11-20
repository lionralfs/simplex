package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func main() {
	// create new example matrix
	maximize := mat.NewVecDense(4, []float64{19, 13, 12, 17})

	constraints := mat.NewDense(3, 5, []float64{3, 2, 1, 2, 225, 1, 1, 1, 1, 117, 4, 3, 3, 4, 420})

	solve(maximize, constraints)
}

func solve(maximize mat.Vector, constraints *mat.Dense) {
	rows, cols := constraints.Dims()
	b := mat.NewVecDense(cols+rows, nil)
	constraintValues := constraints.ColView(cols - 1)
	for i := 0; i < rows; i++ {
		b.SetVec(i, constraintValues.At(i, 0))
	}

	fmt.Printf("maximize vector:\n %v\n\n",
		mat.Formatted(maximize, mat.Prefix(" "), mat.Excerpt(3)))

	A := constraints.Slice(0, rows, 0, cols-1)
	fmt.Printf("A matrix:\n %v\n\n",
		mat.Formatted(A, mat.Prefix(" "), mat.Excerpt(3)))

	B := mat.NewDense(rows, rows, nil)
	for i := 0; i < rows; i++ {
		B.Set(i, i, 1)
	}
	fmt.Printf("B matrix:\n %v\n\n",
		mat.Formatted(B, mat.Prefix(" "), mat.Excerpt(3)))

	fmt.Printf("b vector:\n %v\n\n",
		mat.Formatted(b, mat.Prefix(" "), mat.Excerpt(3)))

	// fmt.Printf("We have %d initial variables.\n", initialVars)
	// fmt.Printf("We have %d constraints.\n", constraints)
	// result := mat.NewDense(rows, cols, nil)

	// for i := 0; i < rows-1; i++ {
	// 	row := problem.RowView(i)
	// 	// set the last element as the first
	// 	result.Set(i, 0, row.At(row.Len()-1, 0))

	// 	for j := 0; j < row.Len()-1; j++ {
	// 		value := row.At(j, 0)
	// 		if i > 0 {
	// 			value *= -1
	// 		}
	// 		result.Set(i, j+1, value)
	// 	}
	// }

	// // display result
	// resultRows, _ := result.Dims()
	// for i := 0; i < resultRows; i++ {
	// 	fmt.Println(result.RawRowView(i))
	// }
}
