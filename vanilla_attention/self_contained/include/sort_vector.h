#pragma once
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

template <typename T>
std::vector<std::size_t> sort_permutation(const std::vector<T> &vec) {
  std::vector<std::size_t> p(vec.size());
  std::iota(p.begin(), p.end(), 0);
  std::sort(p.begin(), p.end(),
            [&](std::size_t i, std::size_t j) { return vec[i] < vec[j]; });
  return p;
}

template <typename T>
void apply_permutation_in_place(std::vector<T> &vec,
                                const std::vector<std::size_t> &p) {
  std::vector<bool> done(vec.size());
  for (std::size_t i = 0; i < vec.size(); ++i) {
    if (done[i]) {
      continue;
    }
    done[i] = true;
    std::size_t prev_j = i;
    std::size_t j = p[i];
    while (i != j) {
      std::swap(vec[prev_j], vec[j]);
      done[j] = true;
      prev_j = j;
      j = p[j];
    }
  }
}

template <typename T>
void sort_vectors_by_row(std::vector<T> &rows, std::vector<T> &cols) {
  auto permutation = sort_permutation(rows);
  apply_permutation_in_place(rows, permutation);
  apply_permutation_in_place(cols, permutation);
}

std::vector<size_t> coo_to_crs(const std::vector<size_t> &coo_rows,
                               const size_t nrows) {
  const size_t nnz = coo_rows.size();
  std::vector<size_t> csr_rows(nrows + 1, 0);
  for (size_t i = 0; i < nnz; i++) {
    csr_rows[coo_rows[i] + 1]++;
  }
  for (size_t i = 0; i < nrows; i++) {
    csr_rows[i + 1] += csr_rows[i];
  }
  return csr_rows;
}
