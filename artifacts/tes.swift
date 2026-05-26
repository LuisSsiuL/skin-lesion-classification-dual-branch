func bubbleSort(_ arr: inout [Int]) {
    let n = arr.count

    for i in 0..<n {
        // Track if a swap was made (optimization)
        var swapped = false

        for j in 0..<(n - i - 1) {
            if arr[j] > arr[j + 1] {
                // Swap elements
                arr.swapAt(j, j + 1)
                swapped = true
            }
        }

        // If no swaps were made, the list is already sorted
        if !swapped {
            break
        }
    }
}

// Example usage
var numbers = [5, 2, 9, 1, 5, 6]
bubbleSort(&numbers)

print("Sorted:", numbers)

