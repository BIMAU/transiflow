#ifndef FVM_FROSCH_PRECONDITIONER_H
#define FVM_FROSCH_PRECONDITIONER_H

#include "Teuchos_RCP.hpp"

#include "Epetra_MultiVector.h"

#include "Ifpack_Preconditioner.h"

#include "Xpetra_CrsMatrix.hpp"

#include "FROSch_Tools_decl.hpp"
#include "FROSch_TwoLevelBlockPreconditioner_def.hpp"


namespace FROSch
{

class Preconditioner: public Ifpack_Preconditioner
{
    typedef KokkosClassic::DefaultNode::DefaultNodeType node_type;
    typedef Xpetra::Matrix<double, int, int> XMatrix;
    typedef Teuchos::RCP<XMatrix> XMatrixPtr;
    typedef Teuchos::RCP<const XMatrix> ConstXMatrixPtr;
    typedef Xpetra::MultiVector<double, int, int> XMultiVector;
    typedef Xpetra::Map<int, int> XMap;
    typedef FROSch::TwoLevelBlockPreconditioner<double, int> FroschPrecType;

public:
    Preconditioner(Teuchos::RCP<const Epetra_RowMatrix> matrix): matrix_(matrix) {}

    virtual ~Preconditioner() {}

    int SetParameters(Teuchos::ParameterList &List) { return 0; }

    int Initialize() { return 0; }

    bool IsInitialized() const { return true; }

    int Compute() { return 0; }

    bool IsComputed() const { return true; }

    double Condest(const Ifpack_CondestType CT = Ifpack_Cheap,
                   const int MaxIters = 1550,
                   const double Tol = 1e-9,
                   Epetra_RowMatrix* Matrix = 0) { return 0.0; }

    double Condest() const { return 0.0; }

    int Apply(const Epetra_MultiVector &X,
              Epetra_MultiVector &Y) const { return -1; }

    int ApplyInverse(const Epetra_MultiVector &X,
                     Epetra_MultiVector &Y) const
        {
            // FIXME: Noop preconditioner
            return Y.Update(1.0, X, 0.0);
        }

    const Epetra_RowMatrix &Matrix() const { return *matrix_; }

    int NumInitialize() const { return 0; }

    int NumCompute() const { return 0; }

    int NumApplyInverse() const { return 0; }

    double InitializeTime() const { return 0; }

    double ComputeTime() const { return 0; }

    double ApplyInverseTime() const { return 0; }

    double InitializeFlops() const { return 0; }

    double ComputeFlops() const { return 0; }

    double ApplyInverseFlops() const { return 0; }

    std::ostream &Print(std::ostream &os) const { return os; }

    int SetUseTranspose(bool UseTranspose) { return -1; }

    bool HasNormInf() const { return false; }

    double NormInf() const { return 0; }

    const char *Label() const { return "FROSchPreconditioner"; }

    bool UseTranspose() const { return false; }

    const Epetra_Comm &Comm() const { return matrix_->Comm(); }

    const Epetra_Map &OperatorDomainMap() const { return matrix_->OperatorDomainMap(); }

    const Epetra_Map &OperatorRangeMap() const { return matrix_->OperatorRangeMap(); }

protected:
    Teuchos::RCP<const Epetra_RowMatrix> matrix_;
};

}

#endif
